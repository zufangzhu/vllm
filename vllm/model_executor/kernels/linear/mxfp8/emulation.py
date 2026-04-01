# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .MXFP8LinearKernel import MXFP8LinearKernel, MXFP8LinearLayerConfig

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_SCALE_DTYPE,
    dequant_mxfp8_to_bf16,
)

import torch


class EmulationMXFP8LinearKernel(MXFP8LinearKernel):
    def get_min_capability(cls) -> int:
        return -1

    def can_implement(cls, c: MXFP8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def __init__(
        self,
        c: MXFP8LinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
    ) -> None:
        super().__init__(c, w_q_param_name, w_s_param_name)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Validate weight_scale dtype and shape (must be 2D for TORCH backend)
        weight_scale = layer.weight_scale
        if weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"TORCH backend requires {MXFP8_SCALE_DTYPE} weight_scale dtype, "
                f"got {weight_scale.dtype}."
            )
        if weight_scale.ndim != 2:
            raise ValueError(
                f"TORCH backend requires 2D weight_scale, got {weight_scale.ndim}D. "
                f"Ensure process_weights_after_loading was called."
            )

        weight_bf16 = dequant_mxfp8_to_bf16(layer.weight, weight_scale)

        output = torch.nn.functional.linear(x, weight_bf16, bias)
        return output.to(self.out_dtype)
