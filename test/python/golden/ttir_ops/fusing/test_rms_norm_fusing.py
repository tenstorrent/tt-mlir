# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from typing import List, Optional

from builder.base.builder_utils import Operand, Shape, get_artifact_dir
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str, dialect: str = "ttnn") -> bool:
    """Check if an op exists in the MLIR file."""
    op_pattern = f"{dialect}.{op_name}"
    with open(mlir_file, "r") as f:
        return any(op_pattern in line for line in f)


def build_torch_rms_norm(
    input_data: torch.Tensor, weight_data: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Build reference RMS norm using PyTorch."""
    x = input_data.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normalized = x / torch.sqrt(variance + epsilon)
    return (weight_data.float() * x_normalized).to(input_data.dtype)


def build_decomposed_rms_norm(
    input: Operand,
    weight: Operand,
    epsilon: float,
    builder: TTIRBuilder,
) -> Operand:
    """
    Build decomposed RMS norm pattern: output = (x / sqrt(mean(x^2) + eps)) * weight
    """
    input_shape = builder.get_shape(input)
    last_dim = input_shape[-1]

    # Typecast to f32 and square
    power_const = builder.constant(torch.full(input_shape, 2.0, dtype=torch.float32))
    input_f32 = builder.typecast(input, output_type=torch.float32)
    squared = builder.pow(input_f32, power_const)

    # Mean, add epsilon, rsqrt
    mean_result = builder.mean(squared, dim_arg=[-1], keep_dim=True)
    reduced_shape = list(input_shape[:-1]) + [1]
    epsilon_const = builder.constant(
        torch.full(reduced_shape, epsilon, dtype=torch.float32)
    )
    variance_eps = builder.add(mean_result, epsilon_const)
    inv_std = builder.rsqrt(variance_eps)

    # Broadcast and normalize
    broadcast_dims = [1] * (len(input_shape) - 1) + [last_dim]
    inv_std_broadcast = builder.broadcast(inv_std, broadcast_dimensions=broadcast_dims)
    normalized = builder.multiply(input_f32, inv_std_broadcast)
    normalized_bf16 = builder.typecast(normalized, output_type=torch.bfloat16)

    # Broadcast weight and apply
    new_weight_shape = [1] * (len(input_shape) - 1) + [last_dim]
    weight_reshaped = builder.reshape(weight, shape=new_weight_shape)
    weight_broadcast_dims = list(input_shape[:-1]) + [1]
    weight_broadcast = builder.broadcast(
        weight_reshaped, broadcast_dimensions=weight_broadcast_dims
    )

    return builder.multiply(weight_broadcast, normalized_bf16)


# =============================================================================
# Main fusion test - covers basic shapes
# =============================================================================
@pytest.mark.parametrize(
    "shape,weight_shape",
    [
        ((32, 2048), (2048,)),
        ((32, 1, 2048), (2048,)),
        ((1, 32, 128, 128), (128,)),
        ((8, 16, 32, 64), (64,)),
    ],
    ids=["2D", "3D", "4D_square", "4D_varied"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("epsilon", [1e-5], ids=["eps_1e5"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rms_norm_fusion(
    shape: Shape,
    weight_shape: Shape,
    dtype: torch.dtype,
    epsilon: float,
    target: str,
    request,
    device,
):
    """Test that decomposed RMS norm pattern is fused into single rms_norm op."""
    shapes = [shape, weight_shape]
    dtypes = [dtype, dtype]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rms_norm_fusion(
            input: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=dtype)
            weight_data = torch.randn(weight_shape, dtype=dtype)
            golden_output = build_torch_rms_norm(input_data, weight_data, epsilon)

            result = build_decomposed_rms_norm(input, weight, epsilon, builder)

            builder.set_goldens(
                {input: input_data, weight: weight_data},
                {result: golden_output},
            )
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )

    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert check_op(output_path, "rms_norm"), "Decomposed RMS norm should be fused"


# =============================================================================
# Fusion with reshape/typecast - tests lookThroughSafeOps tracing
# =============================================================================
def build_rms_norm_with_reshape_and_typecast(
    input: Operand,
    weight: Operand,
    epsilon: float,
    builder: TTIRBuilder,
) -> Operand:
    """Build RMS norm with reshape to 3D and extra typecasts for pattern testing."""
    input_shape = builder.get_shape(input)
    rank = len(input_shape)
    last_dim = input_shape[-1]

    # Compute internal 3D shape by merging all dims except first and last
    if rank == 2:
        merged_dim = 1
    elif rank == 3:
        merged_dim = input_shape[1]
    else:  # 4D+
        merged_dim = 1
        for i in range(1, rank - 1):
            merged_dim *= input_shape[i]

    shape_3d = [input_shape[0], merged_dim, last_dim]

    # Reshape to 3D, then typecast (tests tracing through both)
    input_reshaped = builder.reshape(input, shape=shape_3d)
    input_f32 = builder.typecast(input_reshaped, output_type=torch.float32)

    # Extra typecast to f32 again (no-op but tests pattern)
    input_f32 = builder.typecast(input_f32, output_type=torch.float32)

    # RMS norm computation in 3D
    power_const = builder.constant(torch.full(shape_3d, 2.0, dtype=torch.float32))
    squared = builder.pow(input_f32, power_const)
    mean_result = builder.mean(squared, dim_arg=[-1], keep_dim=True)
    reduced_shape = [input_shape[0], merged_dim, 1]
    epsilon_const = builder.constant(
        torch.full(reduced_shape, epsilon, dtype=torch.float32)
    )
    variance_eps = builder.add(mean_result, epsilon_const)
    inv_std = builder.rsqrt(variance_eps)
    inv_std_broadcast = builder.broadcast(
        inv_std, broadcast_dimensions=[1, 1, last_dim]
    )
    normalized = builder.multiply(input_f32, inv_std_broadcast)
    normalized_bf16 = builder.typecast(normalized, output_type=torch.bfloat16)

    # Weight with extra typecast
    weight_f32 = builder.typecast(weight, output_type=torch.float32)
    weight_bf16 = builder.typecast(weight_f32, output_type=torch.bfloat16)
    weight_reshaped = builder.reshape(weight_bf16, shape=[1, 1, last_dim])
    weight_broadcast = builder.broadcast(
        weight_reshaped, broadcast_dimensions=[input_shape[0], merged_dim, 1]
    )
    output_3d = builder.multiply(weight_broadcast, normalized_bf16)

    # Reshape back to original shape
    return builder.reshape(output_3d, shape=list(input_shape))


@pytest.mark.parametrize(
    "shape,weight_shape",
    [
        ((32, 2048), (2048,)),
        ((8, 64, 256), (256,)),
        ((1, 32, 128, 128), (128,)),
    ],
    ids=["2D", "3D", "4D"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("epsilon", [1e-5], ids=["eps_1e5"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rms_norm_fusion_with_reshape_typecast(
    shape: Shape,
    weight_shape: Shape,
    dtype: torch.dtype,
    epsilon: float,
    target: str,
    request,
    device,
):
    """Test fusion tracing through reshape and typecast ops."""
    shapes = [shape, weight_shape]
    dtypes = [dtype, dtype]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rms_norm_reshape_typecast(
            input: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=dtype)
            weight_data = torch.randn(weight_shape, dtype=dtype)
            golden_output = build_torch_rms_norm(input_data, weight_data, epsilon)

            result = build_rms_norm_with_reshape_and_typecast(
                input, weight, epsilon, builder
            )

            builder.set_goldens(
                {input: input_data, weight: weight_data},
                {result: golden_output},
            )
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )

    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert check_op(
        output_path, "rms_norm"
    ), "RMS norm with reshape/typecast should be fused"


# =============================================================================
# Asymmetric reshape - fuses with compensating reshape
# =============================================================================
def build_rms_norm_asymmetric_reshape(
    input: Operand,
    weight: Operand,
    epsilon: float,
    builder: TTIRBuilder,
) -> Operand:
    """Build RMS norm with reshape on input only (asymmetric): 4D -> 3D."""
    input_shape = builder.get_shape(input)
    last_dim = input_shape[-1]
    # Merge middle dims: (1, 32, 128, 128) -> (1, 4096, 128)
    merged_dim = input_shape[1] * input_shape[2]
    shape_3d = [input_shape[0], merged_dim, last_dim]

    # Reshape 4D to 3D on input
    input_reshaped = builder.reshape(input, shape=shape_3d)
    input_f32 = builder.typecast(input_reshaped, output_type=torch.float32)

    # RMS norm computation in 3D
    power_const = builder.constant(torch.full(shape_3d, 2.0, dtype=torch.float32))
    squared = builder.pow(input_f32, power_const)
    mean_result = builder.mean(squared, dim_arg=[-1], keep_dim=True)
    reduced_shape = [input_shape[0], merged_dim, 1]
    epsilon_const = builder.constant(
        torch.full(reduced_shape, epsilon, dtype=torch.float32)
    )
    variance_eps = builder.add(mean_result, epsilon_const)
    inv_std = builder.rsqrt(variance_eps)
    inv_std_broadcast = builder.broadcast(
        inv_std, broadcast_dimensions=[1, 1, last_dim]
    )
    normalized = builder.multiply(input_f32, inv_std_broadcast)
    normalized_bf16 = builder.typecast(normalized, output_type=torch.bfloat16)

    # Weight broadcast in 3D
    weight_reshaped = builder.reshape(weight, shape=[1, 1, last_dim])
    weight_broadcast = builder.broadcast(
        weight_reshaped, broadcast_dimensions=[input_shape[0], merged_dim, 1]
    )
    output_3d = builder.multiply(weight_broadcast, normalized_bf16)

    # NO reshape back - output stays 3D (asymmetric)
    return output_3d


@pytest.mark.parametrize(
    "shape,weight_shape", [((1, 32, 128, 128), (128,))], ids=["4D_to_3D"]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("epsilon", [1e-5], ids=["eps_1e5"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_rms_norm_asymmetric_reshape(
    shape: Shape,
    weight_shape: Shape,
    dtype: torch.dtype,
    epsilon: float,
    target: str,
    request,
    device,
):
    """Test asymmetric reshape (4D->3D) fuses with compensating reshape."""
    # 3D output: (1, 32*128, 128) = (1, 4096, 128)
    output_shape = (shape[0], shape[1] * shape[2], shape[3])
    shapes = [shape, weight_shape]
    dtypes = [dtype, dtype]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rms_norm_asymmetric(
            input: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=dtype)
            weight_data = torch.randn(weight_shape, dtype=dtype)
            # Golden uses 3D merged shape
            golden_output = build_torch_rms_norm(
                input_data.reshape(output_shape), weight_data, epsilon
            )

            result = build_rms_norm_asymmetric_reshape(input, weight, epsilon, builder)

            builder.set_goldens(
                {input: input_data, weight: weight_data},
                {result: golden_output},
            )
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )

    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert check_op(output_path, "rms_norm") and check_op(
        output_path, "reshape"
    ), "Asymmetric reshape should fuse with compensating reshape"
