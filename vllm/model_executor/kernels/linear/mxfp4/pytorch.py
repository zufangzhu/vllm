# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native PyTorch MXFP4 W4A4 GEMM via ``torch._scaled_mm`` on XPU.

Drives the in-tree PyTorch ``_scaled_mm`` dispatch (oneDNN on XPU)
instead of the vendor-specific ``torch.ops._xpu_C.fp4_gemm`` op.

Both operands are packed FP4 (``float4_e2m1fn_x2``: two 4-bit values per
byte) with ``BlockWise1x32`` E8M0 block scales. oneDNN consumes
*un-swizzled*, *un-padded* scales laid out as ``[outer, ceil_div(K, 32)]``
(``NO_SWIZZLE``).
"""

import torch

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    xpu_mxfp4_quantize as quant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .base import MxFp4LinearKernel, MxFp4LinearLayerConfig

# E8M0 block-scale dtype consumed by torch._scaled_mm for MXFP4 (BlockWise1x32).
_E8M0 = torch.float8_e8m0fnu
_FP4X2 = torch.float4_e2m1fn_x2


class TorchMxFp4LinearKernel(MxFp4LinearKernel):
    """MXFP4 W4A4 GEMM on XPU using the native ``torch._scaled_mm`` dispatch."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "TorchMxFp4 only supports XPU"
        return True, None

    @classmethod
    def can_implement(cls, config: MxFp4LinearLayerConfig) -> tuple[bool, str | None]:
        if config.activation_quant_key != kMxfp4Dynamic:
            return False, "only supports MXFP4 dynamic activation"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # weight: [N, K_packed] packed-fp4; weight_scale: [N, K//32] E8M0.
        # oneDNN wants un-swizzled scales and a column-major weight (B operand).
        weight = layer.weight.view(_FP4X2)
        replace_parameter(layer, "weight", weight.data.t())

        weight_scale = layer.weight_scale.view(_E8M0)
        weight_scale = weight_scale.t().contiguous()
        replace_parameter(layer, "weight_scale", weight_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        weight = layer.weight  # column-major [K_packed, N]
        N = weight.shape[1]

        input_2d = x.reshape(-1, x.shape[-1])
        x_fp4, x_scale = quant_mxfp4(input_2d)
        x_scale = x_scale.view(_E8M0)

        out = torch._scaled_mm(
            x_fp4,
            weight,
            scale_a=x_scale,
            scale_b=layer.weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )

        return out.reshape(*x.shape[:-1], N)
