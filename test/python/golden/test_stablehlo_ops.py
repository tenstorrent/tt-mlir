# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_apis import compile_and_execute_shlo
from test_utils import shape_str, shapes_list_str

pytestmark = pytest.mark.frontend("shlo")


def module_abs(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def abs(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.abs(in0, unit_attrs=unit_attrs)


def module_ceil(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def ceil(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.ceil(in0, unit_attrs=unit_attrs)


def module_cosine(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def cosine(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.cosine(in0, unit_attrs=unit_attrs)


def module_exp(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def exp(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.exp(in0, unit_attrs=unit_attrs)


def module_floor(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def floor(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.floor(in0, unit_attrs=unit_attrs)


def module_neg(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def neg(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.neg(in0, unit_attrs=unit_attrs)


def module_rsqrt(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def rsqrt(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.rsqrt(in0, unit_attrs=unit_attrs)


def module_sine(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def sine(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.sine(in0, unit_attrs=unit_attrs)


def module_sqrt(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def sqrt(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.sqrt(in0, unit_attrs=unit_attrs)


def module_logistic(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def logistic(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.logistic(in0, unit_attrs=unit_attrs)


def module_log(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def log(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.log(in0, unit_attrs=unit_attrs)


def module_and_int(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.int32, torch.int32])
    def and_(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.and_(in0, in1, unit_attrs=unit_attrs)


def module_or_int(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.int32, torch.int32])
    def or_(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.or_(in0, in1, unit_attrs=unit_attrs)


def module_xor_int(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.int32, torch.int32])
    def xor(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.xor(in0, in1, unit_attrs=unit_attrs)


def module_and_bool(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.bool, torch.bool])
    def and_(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.and_(in0, in1, unit_attrs=unit_attrs)


def module_or_bool(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.bool, torch.bool])
    def or_(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.or_(in0, in1, unit_attrs=unit_attrs)


def module_xor_bool(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.bool, torch.bool])
    def xor(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.xor(in0, in1, unit_attrs=unit_attrs)


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


@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_binary_ops(target: str, request, device):
    def module(builder: StableHLOBuilder):
        @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
        def add(
            in0: Operand,
            in1: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.add(in0, in1)

    compile_and_execute_shlo(
        module,
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
        module_abs,
        module_ceil,
        module_cosine,
        module_exp,
        module_floor,
        module_log,
        module_logistic,
        module_neg,
        module_rsqrt,
        module_sine,
        module_sqrt,
    ],
)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    # FP32 sqrt/rsqrt/log fail due to tt-metal untilize NaN handling.
    # See: https://github.com/tenstorrent/tt-metal/pull/33904
    if (
        test_fn.__name__ in ["module_sqrt", "module_rsqrt", "module_log"]
        and dtype == torch.float32
        and target == "ttnn"
    ):
        pytest.xfail(
            f"FP32 {test_fn.__name__} fails due to tt-metal untilize NaN handling. "
            "See: https://github.com/tenstorrent/tt-metal/pull/33904"
        )

    compile_and_execute_shlo(
        test_fn,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


_RESHAPE_CASES = [
    # shapes, semantic id, xfail_ttmetal?
    ([(2, 3), (3, 2)], "swap", True),
    ([(2, 3), (6,)], "flatten", True),
    ([(1, 784), (1, 28, 28)], "unflatten", True),
    ([(4, 8, 16), (4, 128)], "3d_to_2d", True),
    ([(64, 512), (64, 1, 512)], "expand_dims", True),
    ([(128, 128), (64, 256)], "rearrange_2d", True),
    ([(10,), (10,)], "identity", False),
    ([(0, 6), (0, 2, 3)], "zero_dim", True),
]

_RESHAPE_PARAMS = []
for shapes, case_id, xfail_ttmetal in _RESHAPE_CASES:
    # ttnn: expected to pass
    _RESHAPE_PARAMS.append(pytest.param(shapes, "ttnn", id=f"{case_id}-ttnn"))
    # ttmetal: mark as xfail for cases known to be unsupported
    marks = []
    if xfail_ttmetal:
        marks.append(
            pytest.mark.xfail(
                reason="reshape lowering not yet supported in TTMetal backend"
            )
        )
    _RESHAPE_PARAMS.append(
        pytest.param(shapes, "ttmetal", id=f"{case_id}-ttmetal", marks=marks)
    )


@pytest.mark.parametrize("shapes, target", _RESHAPE_PARAMS)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_reshape(
    shapes: tuple[Shape, Shape], target: str, dtype: torch.dtype, request, device
):
    input_shape, output_shape = shapes

    def module(builder: StableHLOBuilder):
        @builder.func([input_shape], [dtype])
        def reshape_wrapper(in0: Operand, builder: StableHLOBuilder):
            if hasattr(builder, "set_graph_level_check"):
                builder.set_graph_level_check(True)
            return builder.reshape(in0, output_shape)

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_reshape_mismatch_raises(target, request, device):
    """
    Element-count mismatch must raise. It may surface as a ValueError from the
    builder or as a TTBuilderCompileException during compile/exec; accept any Exception.
    """
    input_shape = (2, 3)  # 6
    output_shape = (4, 2)  # 8 -> not match

    def module(builder: StableHLOBuilder):
        @builder.func([input_shape], [torch.float32])
        def reshape_wrapper(in0: Operand, builder: StableHLOBuilder):
            if hasattr(builder, "set_graph_level_check"):
                builder.set_graph_level_check(True)
            return builder.reshape(in0, output_shape)

    with pytest.raises(Exception):
        compile_and_execute_shlo(
            module,
            test_base=request.node.name,
            output_root=request.config.getoption("--path"),
            system_desc_path=request.config.getoption("--sys-desc"),
            target=target,
            device=device,
        )


@pytest.mark.parametrize(
    "shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
    [
        (
            [(4, 10, 3, 5, 7), (4, 10, 5, 7, 3)],
            [0],
            [3],
            [0],
            [2],
        )
    ],
)
def test_dot_general(
    shapes: List[Shape],
    batch_dims_lhs: List[int],
    contract_dims_lhs: List[int],
    batch_dims_rhs: List[int],
    contract_dims_rhs: List[int],
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def dot_general(
            in0: Operand,
            in1: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.dot_general(
                in0,
                in1,
                batch_dims_lhs,
                contract_dims_lhs,
                batch_dims_rhs,
                contract_dims_rhs,
            )

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target="ttnn",
        device=device,
    )


# Special handling for tan PCC checks. Due to the vertical asymptote on the tan graph, small changes in input values result in large changes in output values at multiples of pi/2, so both graph and golden tensors must be constrained accordingly.
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_tan(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def tan(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
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
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_ternary_ops(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: StableHLOBuilder):
        @builder.func([shape, shape, shape], [dtype, dtype, dtype])
        def clamp(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.clamp(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module,
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

    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def transpose(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.transpose(in0, permutation, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module,
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
    def module(builder: StableHLOBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def concatenate_wrapper(*inputs_and_builder):
            *inputs, builder = inputs_and_builder
            builder.set_graph_level_check(True)
            return builder.concatenate(list(inputs), dim=dim)

    compile_and_execute_shlo(
        module,
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
    def module(builder: StableHLOBuilder):
        @builder.func(shapes, dtypes)
        def multi_return_model(
            in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder
        ):
            builder.set_graph_level_check(True)

            add_result = builder.add(in0, in1)
            exp_result = builder.exp(in2)
            sqrt_result = builder.sqrt(exp_result)

            return exp_result, sqrt_result

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# Logical operations tests (boolean tensors)
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bool], ids=["bool"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_and_bool,
        module_or_bool,
        module_xor_bool,
    ],
)
def test_logical_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pcc=-1.0,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bool], ids=["bool"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_logical_unary_ops(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module_not_(builder: StableHLOBuilder):
        @builder.func([(128, 128)], [torch.bool])
        def not_(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.not_(in0, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module_not_,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pcc=-1.0,
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
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def slice_fn(in0: Operand, builder: StableHLOBuilder):
            return slice(in0, start_indices, limit_indices, strides, builder)

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# Bitwise operations tests (integer tensors)
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_and_int,
        module_or_int,
        module_xor_int,
    ],
)
def test_bitwise_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_bitwise_unary_ops(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module_not_(builder: StableHLOBuilder):
        @builder.func([(128, 128)], [torch.int32])
        def not_(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.not_(in0, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module_not_,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# ----- Reduce Operations -----


def reduce_sum(
    in0: Operand,
    builder: StableHLOBuilder,
    dimensions: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_sum(in0, dimensions, unit_attrs=unit_attrs)


def reduce_max(
    in0: Operand,
    builder: StableHLOBuilder,
    dimensions: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_max(in0, dimensions, unit_attrs=unit_attrs)


def reduce_min(
    in0: Operand,
    builder: StableHLOBuilder,
    dimensions: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_min(in0, dimensions, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("dimensions", [[0], [1]])
def test_reduce_sum(
    shape: Shape,
    dtype: torch.dtype,
    dimensions: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def reduce_sum_wrapper(in0: Operand, builder: StableHLOBuilder):
            return reduce_sum(in0, builder, dimensions)

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("dimensions", [[0], [1]])
def test_reduce_max(
    shape: Shape,
    dtype: torch.dtype,
    dimensions: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def reduce_max_wrapper(in0: Operand, builder: StableHLOBuilder):
            return reduce_max(in0, builder, dimensions)

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("dimensions", [[0], [1]])
def test_reduce_min(
    shape: Shape,
    dtype: torch.dtype,
    dimensions: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def reduce_min_wrapper(in0: Operand, builder: StableHLOBuilder):
            return reduce_min(in0, builder, dimensions)

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


# ----- Pooling Operations -----


def max_pool_2d(
    in0: Operand,
    builder: StableHLOBuilder,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.pool_2d(
        in0,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_type="max",
        unit_attrs=unit_attrs,
    )


def avg_pool_2d(
    in0: Operand,
    builder: StableHLOBuilder,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.pool_2d(
        in0,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_type="avg",
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize(
    "shape,kernel_size,stride,padding",
    [
        ((32, 32), [3, 3], [2, 2], [1, 1, 1, 1]),
        ((64, 64), [2, 2], [2, 2], [0, 0, 0, 0]),
        ((128, 128), [3, 3], [1, 1], [1, 1, 1, 1]),
        ((1, 32, 64, 64), [1, 1, 3, 3], [1, 1, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1]),
        ((1, 64, 128, 128), [1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0]),
        ((1, 32, 64, 64), [1, 1, 3, 3], [1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]),
    ],
    ids=[
        "rank2_k3s2p1",
        "rank2_k2s2p0",
        "rank2_k3s1p1",
        "rank4_k3s2p1",
        "rank4_k2s2p0",
        "rank4_k3s1p1",
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_max_pool_2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def max_pool_2d_wrapper(in0: Operand, builder: StableHLOBuilder):
            return max_pool_2d(in0, builder, kernel_size, stride, padding)

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shape,kernel_size,stride,padding",
    [
        ((32, 32), [3, 3], [2, 2], [1, 1, 1, 1]),
        ((64, 64), [2, 2], [2, 2], [0, 0, 0, 0]),
        ((128, 128), [3, 3], [1, 1], [1, 1, 1, 1]),
        ((1, 32, 64, 64), [1, 1, 3, 3], [1, 1, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1]),
        ((1, 64, 128, 128), [1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0]),
        ((1, 32, 64, 64), [1, 1, 3, 3], [1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]),
    ],
    ids=[
        "rank2_k3s2p1",
        "rank2_k2s2p0",
        "rank2_k3s1p1",
        "rank4_k3s2p1",
        "rank4_k2s2p0",
        "rank4_k3s1p1",
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_avg_pool_2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def avg_pool_2d_wrapper(in0: Operand, builder: StableHLOBuilder):
            return avg_pool_2d(in0, builder, kernel_size, stride, padding)

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
