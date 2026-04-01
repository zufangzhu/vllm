# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MXFP8LinearLayerConfig,
    choose_mxfp8_linear_kernel,
    EmulationMXFP8LinearKernel,
    FlashInferMXFP8LinearKernel,
)
from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    maybe_post_process_fp8_weight_block,
    process_fp8_weight_block_strategy,
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    QuantMXFP8,
    swizzle_mxfp8_scale,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kFp8StaticTokenSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

__all__ = ["CompressedTensorsW8A8Fp8"]

strategy_to_parameter_type = {
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter,
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter,
    QuantizationStrategy.TENSOR: PerTensorScaleParameter,
}

STATIC_QUANT = True
DYNAMIC_QUANT = False
activation_quant_key_mapping = {
    STATIC_QUANT: kFp8StaticTensorSym,
    DYNAMIC_QUANT: kFp8DynamicTokenSym,
}
weight_quant_key_mapping = {
    QuantizationStrategy.CHANNEL: kFp8StaticTokenSym,
    QuantizationStrategy.TENSOR: kFp8StaticTensorSym,
}
logger = init_logger(__name__)


class CompressedTensorsW8A8MXFp8(CompressedTensorsScheme):
    def __init__(self):
        self.group_size = 32
        self.quant_mxfp8 = QuantMXFP8()

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        if input_size_per_partition % MXFP8_BLOCK_SIZE != 0:
            raise ValueError(
                f"MXFP8 requires input dimension to be divisible by "
                f"{MXFP8_BLOCK_SIZE}, got {input_size_per_partition}"
            )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        # Per Group Weight Scale (MXFP8 uses E8M0 format for scales)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)
        mxfp8_linear_kernel_config = MXFP8LinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=current_platform.fp8_dtype(),
            out_type=self.out_dtype,
        )
        self.mxfp8_linear = choose_mxfp8_linear_kernel(mxfp8_linear_kernel_config)

    def process_weights_after_loading(self, layer) -> None:
        self.mxfp8_linear.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mxfp8_linear.apply_weights(layer, x, bias)
