"""Mamba prefix-cache reuse with the fused XPU GDN kernel (``--linear-attn-backend intel_xpu``).

Under the ``extra_buffer`` strategy, decode copies the GDN state to a track slot at
every ``--mamba-track-interval`` boundary, and a later request with that prefix
restores it. The decode XPU graph is on, so the copy must also run in the captured
fused-kernel graph.

Compare full-vocabulary next-token distributions of the SAME prompt, cold versus
restored from a state that decode tracked. B65 / Qwen3.5-0.8B: without the copy in
the fused path KL = 0.09 to 5.4 (and a different top-1 token); with it KL < 2e-5.
"""

import unittest

import numpy as np
import requests
import torch
from transformers import AutoConfig

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_xpu_ci(est_time=120, suite="stage-b-test-1-gpu-xpu")

TRACK_INTERVAL = 128


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestXPUGDNExtraBufferKL(DefaultServerBase):
    model = "Qwen/Qwen3.5-0.8B"
    other_args = [
        "--skip-tokenizer-init",
        "--attention-backend",
        "intel_xpu",
        "--linear-attn-backend",
        "intel_xpu",
        "--page-size",
        "128",
        "--mamba-track-interval",
        str(TRACK_INTERVAL),
        "--cuda-graph-backend-decode",
        "full",
        "--mem-fraction-static",
        "0.3",
        "--random-seed",
        "42",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        config = AutoConfig.from_pretrained(cls.model)
        cls.vocab_ids = list(range(config.get_text_config().vocab_size))

    def _generate(self, input_ids, max_new_tokens, *, score=False):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": score,
            "logprob_start_len": -1,
        }
        if score:
            payload["token_ids_logprob"] = self.vocab_ids
        response = requests.post(self.base_url + "/generate", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        self.assertEqual(result["meta_info"]["completion_tokens"], max_new_tokens)
        return result

    def _flush_cache(self):
        requests.post(self.base_url + "/flush_cache", timeout=30).raise_for_status()

    def _distribution(self, result):
        rows = result["meta_info"]["output_token_ids_logprobs"]
        self.assertEqual(len(rows), 1)
        self.assertEqual([row[1] for row in rows[0]], self.vocab_ids)
        logprobs = np.asarray([row[0] for row in rows[0]], dtype=np.float64)
        self.assertTrue(np.isfinite(logprobs).all())
        return logprobs - np.logaddexp.reduce(logprobs)

    def test_reuse_distribution_after_decode_tracking(self):
        # 200 prompt + 200 decoded tokens: decode crosses the 256 and 384
        # boundaries, so the cached state at 384 was written during decode.
        for trial in range(2):
            with self.subTest(trial=trial):
                seed_prompt = [
                    (1000 + 37 * trial + 13 * i) % 30000 + 100 for i in range(200)
                ]
                self._flush_cache()
                generated = self._generate(seed_prompt, 200)["output_ids"]
                prompt = seed_prompt + generated[:190]

                self._flush_cache()
                reference = self._generate(prompt, 1, score=True)
                reference_logprobs = self._distribution(reference)

                self._flush_cache()
                self._generate(seed_prompt, 200)
                reused = self._generate(prompt, 1, score=True)
                self.assertEqual(
                    reused["meta_info"]["cached_tokens"], 3 * TRACK_INTERVAL
                )
                reused_logprobs = self._distribution(reused)

                kl = float(
                    np.sum(
                        np.exp(reference_logprobs)
                        * (reference_logprobs - reused_logprobs)
                    )
                )
                print(f"trial {trial}: KL(reference || reuse)={kl:.9g}", flush=True)
                self.assertGreaterEqual(kl, -1e-6)
                self.assertLess(
                    kl,
                    1e-4,
                    "Restoring a decode-tracked state changed the distribution",
                )


if __name__ == "__main__":
    unittest.main()
