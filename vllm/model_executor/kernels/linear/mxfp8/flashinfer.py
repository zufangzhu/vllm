# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .MXFP8LinearKernel import MXFP8LinearKernel, MXFP8LinearLayerConfig

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.utils import flashinfer as vllm_flashinfer

import torch


class FlashInferMXFP8LinearKernel(MXFP8LinearKernel):
    def get_min_capability(cls) -> int:
        return 80

    def can_implement(cls, c: MXFP8LinearLayerConfig) -> tuple[bool, str | None]:
        # TODO
        if c.weight_type != torch.float8_e4m3fn:
            return False, "FlashInfer MXFP8 Linear only supports FP8 (e4m3) weights"

        in_features, out_features = c.partition_weight_shape
        if in_features % MXFP8_BLOCK_SIZE or out_features % MXFP8_BLOCK_SIZE:
            return (
                False,
                f"FlashInfer MXFP8 Linear requires in/out features to be multiples of {MXFP8_BLOCK_SIZE}, "
                f"got in_features={in_features}, out_features={out_features}",
            )

        return True, None

    def __init__(
        self,
        c: MXFP8LinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
    ) -> None:
        super().__init__(c, w_q_param_name, w_s_param_name)
        # Minimum dimension size for F8_128x4 block scaling layout
        self.min_dim = 128

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, K = weight.shape

        input_shape = x.shape
        input_2d = x.view(-1, K)
        M_orig = input_2d.shape[0]

        assert self.min_dim <= K, (
            f"mm_mxfp8 requires K >= {self.min_dim}, got K={K}. "
            f"in_features is too small for mm_mxfp8."
        )
        assert self.min_dim <= N, (
            f"mm_mxfp8 requires N >= {self.min_dim}, got N={N}. "
            f"out_features is too small for mm_mxfp8."
        )

        M_padded = ((M_orig + self.min_dim - 1) // self.min_dim) * self.min_dim
        if M_padded != M_orig:
            pad_rows = M_padded - M_orig
            input_2d = torch.nn.functional.pad(input_2d, (0, 0, 0, pad_rows))

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input_2d,
            is_sf_swizzled_layout=True,  # Swizzled for best accuracy
        )

        if not weight.is_contiguous():
            weight = weight.contiguous()

        output = vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            layer.weight_scale,
            out_dtype=self.out_dtype,
            backend="cutlass",
        )

        if M_padded != M_orig:
            output = output[:M_orig, :]

        if bias is not None:
            output = output + bias

        output_shape = (*input_shape[:-1], N)
        return output.view(output_shape)
