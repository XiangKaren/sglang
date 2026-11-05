"""`--load-format layered` for models without `load_weights_to_module`: modules
loaded in host memory are moved to the device one at a time and post-processed."""

import unittest

import torch
from torch import nn

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_loader.loader import (
    _host_tensor_attributes,
    _move_modules_and_postprocess,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


class RecordingQuantMethod:
    """Stands in for online quantization: replaces the weight after loading."""

    def __init__(self, calls: list, device: torch.device):
        self.calls = calls
        self.device = device

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        on_device = all(
            t.device.type == self.device.type
            for t in list(layer.parameters()) + list(layer.buffers())
        )
        self.calls.append((layer.name, on_device))
        layer.weight = nn.Parameter(layer.weight.data * 2, requires_grad=False)


class QuantizedLinear(nn.Linear):
    def __init__(self, name: str, quant_method: RecordingQuantMethod):
        super().__init__(4, 4, bias=False)
        self.name = name
        self.quant_method = quant_method


class GatedDeltaNetLike(nn.Module):
    """Caches a view of its convolution weight, as Qwen3.5 GDN layers do."""

    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(3, 3, kernel_size=3, groups=3, bias=False)
        self.attn = RadixLinearAttention(
            layer_id=0,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=1,
            head_k_dim=1,
            head_v_dim=1,
            conv_weights=self.conv1d.weight.view(3, 3),
        )


class TestLayeredLoaderHostStaging(CustomTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.accelerator.current_accelerator(check_available=True)
        if self.device is None:
            self.skipTest("An accelerator is required to exercise host staging")

    def test_moves_all_state_and_postprocesses_children_first(self) -> None:
        calls = []
        quant_method = RecordingQuantMethod(calls, self.device)
        parent = QuantizedLinear("parent", quant_method)
        parent.child = QuantizedLinear("child", quant_method)
        parent.register_buffer("scale", torch.ones(4))
        weights = {
            name: module.weight.detach().clone()
            for name, module in (("parent", parent), ("child", parent.child))
        }

        _move_modules_and_postprocess(parent, self.device)

        # The child runs first, and each module is fully on the device when its
        # own processing runs.
        self.assertEqual(calls, [("child", True), ("parent", True)])
        for tensor in list(parent.parameters()) + list(parent.buffers()):
            self.assertEqual(tensor.device.type, self.device.type)
        for name, module in (("parent", parent), ("child", parent.child)):
            torch.testing.assert_close(module.weight.cpu(), weights[name] * 2)

    def test_rebuilds_cached_convolution_view(self) -> None:
        model = GatedDeltaNetLike()

        _move_modules_and_postprocess(model, self.device)

        conv_weights = model.attn.conv_weights
        self.assertEqual(conv_weights.device.type, self.device.type)
        self.assertEqual(conv_weights.data_ptr(), model.conv1d.weight.data_ptr())

    def test_gemma_norm_buffer_follows_its_weight(self) -> None:
        norm = GemmaRMSNorm(8)
        loaded = torch.randn(8)
        norm.weight.weight_loader(norm.weight, loaded)

        _move_modules_and_postprocess(norm, self.device)

        self.assertEqual(norm.gemma_weight.device.type, self.device.type)
        torch.testing.assert_close(norm.gemma_weight.cpu(), loaded + 1.0)

    def test_reports_plain_tensors_left_on_host(self) -> None:
        model = nn.Module()
        model.layer = nn.Linear(4, 4)
        model.layer.lookup = torch.zeros(4)

        _move_modules_and_postprocess(model, self.device)

        self.assertEqual(_host_tensor_attributes(model), ["layer.lookup"])


if __name__ == "__main__":
    unittest.main()
