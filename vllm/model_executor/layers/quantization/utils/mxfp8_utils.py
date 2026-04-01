# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

from vllm.logger import init_logger
from vllm import _custom_ops as ops
from vllm.model_executor.custom_op import CustomOp
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.platforms import current_platform

logger = init_logger(__name__)


# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Swizzle MXFP8 scales from row-major 2D to F8_128x4 layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    sf_swizzled = sf_reshaped.transpose(1, 3)

    return sf_swizzled.contiguous().view(-1)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 tensor to BF16."""
    x_float = x.to(torch.float32)

    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    dequantized = x_blocked * descale.unsqueeze(-1)

    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


@CustomOp.register("quant_mxfp8")
class QuantMXFP8(CustomOp):
    """
    Quantize input tensor to MXFP8
    This CustomOp supports both static and dynamic quantization.
    """

    def __init__(
        self,
    ):
        pass

    def forward_cuda(
        self, x: torch.Tensor, is_sf_swizzled_layout: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

        x_q, x_scales = flashinfer_mxfp8_quantize(
            x, is_sf_swizzled_layout=is_sf_swizzled_layout
        )
        if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
            x_scales = x_scales.view(x.size(0), -1)
        return x_q, x_scales

    def forward_xpu(
        self, x: torch.Tensor, is_sf_swizzled_layout: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mxfp8_dtype = current_platform.fp8_dtype()
        finfo = torch.finfo(mxfp8_dtype)
        fp8_min = finfo.min
        fp8_max = finfo.max
        eps = 1e-10
        x_q = torch.empty_like(x, device=x.device, dtype=mxfp8_dtype)
        shape = x.shape[:-1] + (x.shape[-1] // MXFP8_BLOCK_SIZE,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
        torch.ops._C.per_token_group_fp8_quant(
            x, x_q, x_s, MXFP8_BLOCK_SIZE, eps, fp8_min, fp8_max, True
        )
        x_s = x_s.to(torch.float8_e8m0fnu)
        return x_q, x_s

    def forward_native(
        self, x: torch.Tensor, is_sf_swizzled_layout: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fake implementation for torch.compile tracing."""
        fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

        block_size = MXFP8_BLOCK_SIZE

        if x.ndim == 2:
            M, N = x.shape
            K = (N + block_size - 1) // block_size
            if is_sf_swizzled_layout:
                M_padded = ((M + 127) // 128) * 128
                K_padded = ((K + 3) // 4) * 4
                scales = torch.empty(
                    M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
                )
            else:
                scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
        elif x.ndim == 3:
            B, M, N = x.shape
            K = (N + block_size - 1) // block_size
            if is_sf_swizzled_layout:
                M_padded = ((M + 127) // 128) * 128
                K_padded = ((K + 3) // 4) * 4
                scales = torch.empty(
                    B * M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
                )
            else:
                scales = torch.empty(
                    (B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device
                )
        else:
            scale_shape = list(x.shape)
            scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
            scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

        return fp_data, scales
