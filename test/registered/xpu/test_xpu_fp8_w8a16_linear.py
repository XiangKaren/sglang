"""Unit tests for online weight-only FP8 (W8A16) dense linear on XPU."""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

DEV = "xpu"
M_VALUES = (1, 8, 300)


def _make_layer(weight: torch.Tensor):
    """A bare ``LinearBase`` holding a bf16 weight, as after online loading."""
    from sglang.srt.layers.linear import LinearBase

    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return layer


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def _w8a16_skip_reason():
    from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
        _has_fp8_w8a16_gemm,
        _is_xe2,
    )

    if not _is_xe2():
        return "W8A16 dense linear is used on Xe2 only"
    if not _has_fp8_w8a16_gemm():
        return "sgl_kernel has no fp8_w8a16_gemm"
    return None


@unittest.skipUnless(is_xpu(), "requires XPU")
class TestXPUFp8W8A16Linear(CustomTestCase):
    """The W8A16 path itself: runs on Xe2 with a kernel that has fp8_w8a16_gemm."""

    @classmethod
    def setUpClass(cls):
        reason = _w8a16_skip_reason()
        if reason:
            raise unittest.SkipTest(reason)

    def _method(self):
        from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
            XPUFp8W8A16LinearMethod,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        return XPUFp8W8A16LinearMethod(Fp8Config())

    def test_dispatch(self):
        from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
            XPUFp8W8A16LinearMethod,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        layer = _make_layer(torch.empty(64, 64, dtype=torch.bfloat16))
        online = Fp8Config().get_quant_method(layer, prefix="")
        self.assertIsInstance(online, XPUFp8W8A16LinearMethod)

        for config in (
            Fp8Config(is_checkpoint_fp8_serialized=True),
            Fp8Config(use_mxfp8=True),
        ):
            method = config.get_quant_method(layer, prefix="")
            self.assertNotIsInstance(method, XPUFp8W8A16LinearMethod)

    def _check_numerics(self, n: int, k: int, expect_scale_shape):
        torch.manual_seed(0)
        weight = torch.randn(n, k, dtype=torch.bfloat16, device=DEV) * 0.02
        bias = torch.randn(n, dtype=torch.bfloat16, device=DEV)
        layer = _make_layer(weight.clone())
        method = self._method()
        method.process_weights_after_loading(layer)

        self.assertTrue(layer.use_fp8_w8a16)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.weight.shape), (n, k))
        self.assertEqual(tuple(layer.weight_scale.shape), expect_scale_shape)

        for m in M_VALUES:
            x = torch.randn(m, k, dtype=torch.bfloat16, device=DEV)
            out = method.apply(layer, x, bias)
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(tuple(out.shape), (m, n))
            self.assertLess(_rel_err(out, F.linear(x, weight, bias)), 0.05)

    def test_block_scales(self):
        self._check_numerics(n=576, k=2048, expect_scale_shape=(5, 16))

    def test_per_tensor_scale_when_k_not_block_aligned(self):
        self._check_numerics(n=256, k=160, expect_scale_shape=())

    def test_3d_input(self):
        weight = torch.randn(256, 256, dtype=torch.bfloat16, device=DEV) * 0.02
        layer = _make_layer(weight.clone())
        method = self._method()
        method.process_weights_after_loading(layer)

        x = torch.randn(2, 5, 256, dtype=torch.bfloat16, device=DEV)
        out = method.apply(layer, x)
        self.assertEqual(tuple(out.shape), (2, 5, 256))
        self.assertLess(_rel_err(out, F.linear(x, weight)), 0.05)

    def test_unaligned_layer_stays_bf16(self):
        # Linear-attention gate projections (N = 32) do not fit the kernel.
        weight = torch.randn(32, 2048, dtype=torch.bfloat16, device=DEV)
        layer = _make_layer(weight.clone())
        method = self._method()
        method.process_weights_after_loading(layer)

        self.assertFalse(layer.use_fp8_w8a16)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        x = torch.randn(4, 2048, dtype=torch.bfloat16, device=DEV)
        torch.testing.assert_close(method.apply(layer, x), F.linear(x, weight))


@unittest.skipUnless(is_xpu(), "requires XPU")
class TestXPUFp8W8A16Dispatch(CustomTestCase):
    """Per-architecture dispatch with the capability mocked, so both sides of the
    gate are checked on any XPU runner."""

    def test_dispatch_selects_w8a16_on_xe2(self):
        from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
            XPUFp8W8A16LinearMethod,
            _has_fp8_w8a16_gemm,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.platforms import current_platform
        from sglang.srt.platforms.device_mixin import DeviceCapability

        if not _has_fp8_w8a16_gemm():
            self.skipTest("sgl_kernel has no fp8_w8a16_gemm")
        layer = _make_layer(torch.empty(64, 64, dtype=torch.bfloat16))
        with mock.patch.object(
            current_platform,
            "get_device_capability",
            return_value=DeviceCapability(2, 0),
        ):
            method = Fp8Config().get_quant_method(layer, prefix="")
        self.assertIsInstance(method, XPUFp8W8A16LinearMethod)

    def test_dispatch_keeps_w8a8_on_native_fp8_arch(self):
        # Architectures with native FP8 matrix support (Xe3P reports 3.5) keep
        # the W8A8 Fp8LinearMethod path.
        from sglang.srt.hardware_backend.xpu.quantization.fp8_w8a16 import (
            XPUFp8W8A16LinearMethod,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
        from sglang.srt.platforms import current_platform
        from sglang.srt.platforms.device_mixin import DeviceCapability

        layer = _make_layer(torch.empty(64, 64, dtype=torch.bfloat16))
        with mock.patch.object(
            current_platform,
            "get_device_capability",
            return_value=DeviceCapability(3, 5),
        ):
            method = Fp8Config().get_quant_method(layer, prefix="")
        self.assertIsInstance(method, Fp8LinearMethod)
        self.assertNotIsInstance(method, XPUFp8W8A16LinearMethod)


if __name__ == "__main__":
    unittest.main()
