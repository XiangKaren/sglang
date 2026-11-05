"""Online FP8 GSM8K on Intel XPU with `--load-format layered`.

The model has no `load_weights_to_module`, so it is loaded in host memory and moved
to the device one module at a time, each quantized before the next is moved.
Radix cache and overlap scheduling stay at their defaults.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=300, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestXPULayeredLoadOnlineFp8GSM8K(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    # 0.66 in each of 3 runs (B65). The default loader scores 0.61 on the same setup:
    # the RoPE cos/sin table is built on the host here, and that last-bit difference
    # alone moves this score, so the threshold sits below both.
    accuracy = 0.60
    num_examples = 100
    other_args = [
        "--device",
        "xpu",
        "--attention-backend",
        "intel_xpu",
        "--dtype",
        "bfloat16",
        "--quantization",
        "fp8",
        "--load-format",
        "layered",
        "--mem-fraction-static",
        "0.6",
    ]


if __name__ == "__main__":
    unittest.main()
