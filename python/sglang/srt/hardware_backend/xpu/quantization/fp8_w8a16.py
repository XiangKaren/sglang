# SPDX-License-Identifier: Apache-2.0
"""Online weight-only FP8 (W8A16) dense linear for Intel XPU (Xe2).

Xe2 has no FP8 matrix engine, so instead of quantizing activations for
``torch._scaled_mm`` (W8A8), the weights are quantized to FP8 at load time and
the activations stay in bf16. The GEMM is sgl-kernel-xpu's W8A16 kernel, the
same one the XPU FP8 MoE path already runs for its experts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import (
    input_to_float8,
    per_block_cast_to_fp8,
)
from sglang.srt.platforms import current_platform

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.fp8 import Fp8Config

# Shape limits of the W8A16 kernel; layers outside them stay in bf16.
_N_ALIGN = 64
_K_ALIGN = 32
_BLOCK = 128


def _has_fp8_w8a16_gemm() -> bool:
    try:
        from sgl_kernel import fp8_w8a16_gemm  # noqa: F401
    except ImportError:
        return False
    return True


def _is_xe2() -> bool:
    # Xe2 (Battlemage) reports compute capability 2.x. Later architectures with
    # native FP8 matrix support (Xe3P reports 3.5) keep the W8A8 path.
    return current_platform.get_device_capability().major == 2


def use_xpu_fp8_w8a16(quant_config: Fp8Config) -> bool:
    """Whether online FP8 dense layers on XPU should run weight-only W8A16."""
    return (
        not quant_config.is_checkpoint_fp8_serialized
        and not quant_config.use_mxfp8
        and quant_config.activation_scheme == "dynamic"
        and _is_xe2()
        and _has_fp8_w8a16_gemm()
    )


class XPUFp8W8A16LinearMethod(Fp8LinearMethod):
    """Fp8LinearMethod for online FP8 on XPU: FP8 weights, bf16 activations.

    Weights are loaded in bf16 (``create_weights`` is inherited) and quantized
    after loading: with 128x128 block scales when K is a multiple of 128,
    otherwise with one per-tensor scale.
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight.data
        n, k = weight.shape
        if weight.dtype != torch.bfloat16 or n % _N_ALIGN != 0 or k % _K_ALIGN != 0:
            # The kernel needs bf16 activations and aligned shapes; small
            # projections such as linear-attention gates fall back to bf16.
            layer.use_fp8_w8a16 = False
            layer.weight = Parameter(weight, requires_grad=False)
            return

        if k % _BLOCK == 0:
            qweight, weight_scale = per_block_cast_to_fp8(weight)
        else:
            qweight, weight_scale = input_to_float8(weight)
        layer.weight = Parameter(qweight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.float(), requires_grad=False)
        layer.input_scale = None
        layer.use_fp8_w8a16 = True

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not layer.use_fp8_w8a16:
            return F.linear(x, layer.weight, bias)

        from sgl_kernel import fp8_w8a16_gemm

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        output = fp8_w8a16_gemm(
            x_2d,
            layer.weight,
            layer.weight_scale,
            bias=None if bias is None else bias.float(),
        )
        return output.view(*x.shape[:-1], output.shape[-1])
