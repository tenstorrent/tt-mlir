# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple

from builder.base.builder import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import compile_and_execute_shlo
from test_utils import shape_str

pytestmark = pytest.mark.frontend("shlo")


def add(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.add(in0, in1, unit_attrs=unit_attrs)


def clamp(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.clamp(in0, in1, in2, unit_attrs=unit_attrs)


def abs(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.abs(in0, unit_attrs=unit_attrs)


def ceil(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.ceil(in0, unit_attrs=unit_attrs)


def cosine(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.cosine(in0, unit_attrs=unit_attrs)


def exp(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.exp(in0, unit_attrs=unit_attrs)


def floor(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.floor(in0, unit_attrs=unit_attrs)


def neg(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.neg(in0, unit_attrs=unit_attrs)


def rsqrt(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.rsqrt(in0, unit_attrs=unit_attrs)


def sine(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.sine(in0, unit_attrs=unit_attrs)


def sqrt(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.sqrt(in0, unit_attrs=unit_attrs)


def logistic(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.logistic(in0, unit_attrs=unit_attrs)


def log(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.log(in0, unit_attrs=unit_attrs)


def slice(
    in0: Operand,
    start_indices: List[int],
    limit_indices: List[int],
    strides: Optional[List[int]],
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.slice(
        in0,
        start_indices=start_indices,
        limit_indices=limit_indices,
        strides=strides,
        unit_attrs=unit_attrs,
    )


def transpose(
    in0: Operand,
    builder: StableHLOBuilder,
    permutation: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.transpose(in0, permutation, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        add,
    ],
)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        abs,
        ceil,
        cosine,
        exp,
        floor,
        log,
        logistic,
        neg,
        rsqrt,
        sine,
        sqrt,
    ],
)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# Special handling for tan PCC checks. Due to the vertical asymptote on the tan graph, small changes in input values result in large changes in output values at multiples of pi/2, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_tan(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def tan(
        in0: Operand, builder: StableHLOBuilder, unit_attrs: Optional[List[str]] = None
    ):
        import math

        randn_tensor = torch.randn(shape, dtype=dtype)
        input_golden = randn_tensor.uniform_(
            (-math.pi / 2 + 0.05), (math.pi / 2 - 0.05)
        )
        output_golden = torch.tan(input_golden)
        tan_0 = builder.tan(in0, unit_attrs=unit_attrs)
        builder.set_goldens({in0: input_golden}, {tan_0: output_golden})
        builder.set_graph_level_check(True)
        return tan_0

    compile_and_execute_shlo(
        tan,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        clamp,
    ],
)
def test_ternary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        [shape, shape, shape],
        [dtype, dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(2, 3, 4), (128, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "permutation",
    [
        [1, 0],
        [2, 1, 0],
        [0, 2, 1],
    ],
)
def test_transpose(
    shape: Shape,
    dtype: torch.dtype,
    permutation: List[int],
    target: str,
    request,
    device,
):
    if len(shape) != len(permutation):
        pytest.skip(f"Permutation {permutation} doesn't match shape rank {len(shape)}")

    # Skip ttmetal for dimensions > 2
    if target == "ttmetal" and len(shape) > 2:
        pytest.skip(
            f"ttmetal does not support transpose for dimensions > 2, got shape with {len(shape)} dimensions"
        )

    compile_and_execute_shlo(
        lambda in0, builder: transpose(in0, builder, permutation),
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shapes,dim",
    [
        ([(64, 128), (64, 128)], 0),  # 2 tensors, dim 0
        ([(128, 64), (128, 64)], 1),  # 2 tensors, dim 1
        ([(64, 128), (32, 128), (16, 128)], 0),  # 3 tensors, dim 0
        ([(32, 64), (32, 128)], 1),  # Different sizes in dim
        ([(64, 64), (64, 64), (64, 64)], 0),  # 3 identical tensors
        ([(128, 64), (128, 64), (128, 64), (128, 64)], 1),  # 4 tensors
    ],
    ids=["2t_dim0", "2t_dim1", "3t_dim0_ttir", "diff_size", "3t_same", "4t_dim1"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_concatenate(
    shapes: List[Shape],
    dim: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    # Create a wrapper function
    def concatenate_wrapper(*inputs_and_builder):
        *inputs, builder = inputs_and_builder
        builder.set_graph_level_check(True)
        return builder.concatenate(list(inputs), dim=dim)

    # Set the name for better test identification.
    concatenate_wrapper.__name__ = "concatenate"

    compile_and_execute_shlo(
        concatenate_wrapper,
        shapes,
        [dtype] * len(shapes),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shapes", [[(64, 64), (64, 64), (64, 64)]], ids=["64x64"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_stablehlo_multi_return_support(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    def multi_return_model(
        in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder
    ):
        builder.set_graph_level_check(True)

        add_result = builder.add(in0, in1)
        exp_result = builder.exp(in2)
        sqrt_result = builder.sqrt(exp_result)

        return exp_result, sqrt_result

    compile_and_execute_shlo(
        multi_return_model,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shape,start_indices,limit_indices,strides",
    [
        ((128, 128), [0, 0], [64, 64], [1, 1]),
        ((128, 128), [32, 32], [96, 96], [1, 1]),
        ((128, 128), [0, 0], [128, 64], [2, 1]),
        ((256, 256), [64, 64], [192, 192], [1, 1]),
    ],
    ids=["128x128_basic", "128x128_offset", "128x128_stride", "256x256_large"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_slice(
    shape: Shape,
    start_indices: List[int],
    limit_indices: List[int],
    strides: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def slice_fn(in0: Operand, builder: StableHLOBuilder):
        return slice(in0, start_indices, limit_indices, strides, builder)

    compile_and_execute_shlo(
        slice_fn,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
