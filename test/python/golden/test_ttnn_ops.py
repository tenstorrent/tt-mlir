# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple

from builder.base.builder import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_utils import compile_and_execute_ttnn
from test_utils import shape_str, shapes_list_str
from op_wrappers.eltwise import *

pytestmark = pytest.mark.frontend("ttnn")


unary_ops = [
    abs,
    atan,
    cbrt,
    ceil,
    cos,
    erf,
    erfc,
    exp,
    expm1,
    floor,
    gelu,
    isfinite,
    log,
    log1p,
    logical_not,
    neg,
    reciprocal,
    relu,
    relu6,
    rsqrt,
    sigmoid,
    sign,
    silu,
    sin,
    sqrt,
    tan,
    tanh,
    mish,
]
unary_ops_dtypes = [
    torch.float32,
    torch.int32,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", unary_ops_dtypes, ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
@pytest.mark.parametrize("test_fn", unary_ops)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if dtype == torch.int32 and test_fn not in [
        abs,
        neg,
        relu,
    ]:
        pytest.skip("int32 unary op is not supported yet for this operation")
    if test_fn == mish and dtype == torch.float32:
        pytest.xfail(
            "Mish with float 32 causes PCC: https://github.com/tenstorrent/tt-metal/issues/31112"
        )

    pipeline_options = []
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


bitwise_unary_ops = [bitwise_not]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", bitwise_unary_ops)
def test_bitwise_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


unary_ops_with_float_param = [leaky_relu]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", unary_ops_with_float_param)
@pytest.mark.parametrize("parameter", [0.01, 0.1, 0.2])
def test_unary_ops_with_float_param(
    test_fn: Callable,
    parameter: float,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def wrapper_func(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return test_fn(in0, parameter, builder, unit_attrs=unit_attrs)

    pipeline_options = []
    compile_and_execute_ttnn(
        wrapper_func,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
        pcc=0.98,
    )


# Create hoisted versions of operations by currying the unit_attrs parameter
def create_hoisted_unary_op(op_func, name):
    """Create a hoisted version of a unary operation by adding the should_hoist unit attribute"""

    def hoisted_op(in0, builder, **kwargs):
        # For unary ops
        return op_func(in0, builder, unit_attrs=["ttnn.should_hoist"], **kwargs)

    # Set the name for better test identification
    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


hoisted_unary_ops = [
    create_hoisted_unary_op(exp, "exp"),
    create_hoisted_unary_op(abs, "abs"),
    create_hoisted_unary_op(ceil, "ceil"),
    create_hoisted_unary_op(floor, "floor"),
    create_hoisted_unary_op(tanh, "tanh"),
    create_hoisted_unary_op(reciprocal, "reciprocal"),
    create_hoisted_unary_op(neg, "neg"),
    create_hoisted_unary_op(sigmoid, "sigmoid"),
    create_hoisted_unary_op(sin, "sin"),
    create_hoisted_unary_op(cos, "cos"),
    create_hoisted_unary_op(relu, "relu"),
]


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("test_fn", hoisted_unary_ops)
def test_cpu_hoistable_unary_ops(
    test_fn: Callable,
    shape: Shape,
    request,
    device,
    dtype: torch.dtype = torch.float32,
):
    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=f"{request.node.name}",
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


binary_ops = [
    add,
    atan2,
    divide,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    remainder,
    subtract,
    pow_tensor,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", binary_ops)
def test_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    pipeline_options = []
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape, shape],
        inputs_types=[dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


binary_bitwise_ops = [
    bitwise_and,
    bitwise_or,
    bitwise_xor,
]
binary_bitwise_dtypes = [
    torch.int32,
    torch.uint32,
    torch.uint16,
    torch.uint8,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", binary_bitwise_dtypes, ids=["i32", "u32", "u16", "u8"]
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", binary_bitwise_ops)
def test_bitwise_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape, shape],
        inputs_types=[dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


binary_logical_shift_ops = [
    logical_left_shift,
    logical_right_shift,
]
binary_logical_shift_dtypes = [
    torch.int32,
    torch.uint32,
    torch.uint16,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", binary_logical_shift_dtypes, ids=["i32", "u32", "u16"]
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", binary_logical_shift_ops)
def test_logical_shift_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    if test_fn == logical_left_shift and dtype == torch.uint16:
        pytest.xfail("uint16 logical left shift op is not supported yet")
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape, shape],
        inputs_types=[dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


binary_comparison_ops = [
    eq,
    ge,
    gt,
    le,
    lt,
    ne,
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int32], ids=["f32", "bf16", "i32"]
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", binary_comparison_ops)
def test_comparison_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def comparison_ops(
        in0: Operand,
        in1: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        randn_tensor1 = torch.randn(shape, dtype=torch.float32)
        randn_tensor2 = torch.randn(shape, dtype=torch.float32)

        # Set some indices in randn_tensor2 to be the same as randn_tensor1
        # This ensures we have both equal and unequal values for comprehensive testing
        num_elements = torch.numel(randn_tensor1)
        num_equal_indices = num_elements // 2

        equal_indices = torch.randperm(num_elements)[:num_equal_indices]
        randn_tensor2.view(-1)[equal_indices] = randn_tensor1.view(-1)[equal_indices]

        input_tensor1 = randn_tensor1.to(dtype)
        input_tensor2 = randn_tensor2.to(dtype)

        builder.set_goldens(inputs={in0: input_tensor1, in1: input_tensor2})

        return test_fn(in0, in1, builder, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        comparison_ops,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


ternary_ops = [
    where | Marks(pytest.mark.xfail(reason="Fails Golden")),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", ternary_ops)
def test_ternary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    pipeline_options = []
    compile_and_execute_ttnn(
        test_fn,
        inputs_shapes=[shape, shape, shape],
        inputs_types=[dtype, dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


# Ternary eltwise ops with implicit broadcasting
@pytest.mark.xfail(reason="Fails Golden")
@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 16, 32), (8, 16, 32), (8, 16, 32)],
        [(8, 16, 32), (1, 16, 32), (8, 16, 32)],
        [(8, 16, 32), (8, 16, 32), (1, 16, 32)],
        [(8, 16, 32), (1, 1, 32), (1, 1, 32)],
        [(1, 1, 32), (8, 16, 32), (1, 1, 32)],
        [(1, 1, 32), (1, 1, 32), (8, 16, 32)],
        [(1, 16, 32), (8, 1, 32), (8, 16, 1)],
        [(1, 4, 1), (1, 4, 768), (1, 1, 1)],
        [(1, 1, 1, 4), (1, 1, 1, 1), (1, 1, 1, 1)],
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize(
    "input_dtypes",
    [
        pytest.param((torch.float32, torch.float32, torch.float32), id="f32-f32-f32"),
        pytest.param((torch.float32, torch.int32, torch.int32), id="f32-i32-i32"),
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", [where])
def test_ternary_eltwise_ops_implicit_broadcast(
    test_fn: Callable,
    shapes: List[Shape],
    input_dtypes: Tuple[torch.dtype, torch.dtype, torch.dtype],
    target: str,
    request,
    device,
):
    dtype1, dtype2, dtype3 = input_dtypes

    compile_and_execute_ttnn(
        test_fn,
        shapes,
        [dtype1, dtype2, dtype3],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("max_arg,min_arg", [(3.0, 2.0)])
def test_clamp_scalar(shape: Shape, max_arg: float, min_arg: float, request, device):
    def clamp_scalar(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        print(f"Clamping with min: {min_arg}, max: {max_arg}")
        return builder.clamp_scalar(
            in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
        )

    compile_and_execute_ttnn(
        clamp_scalar,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes", [[(32, 64), (32, 64), (32, 64)]], ids=shapes_list_str
)
def test_clamp_tensor(shapes: List[Shape], request, device):
    def clamp_tensor(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.clamp_tensor(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        clamp_tensor,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes", [[(10, 64, 32), (32, 128), (1,)]], ids=shapes_list_str
)
def test_linear(shapes: List[Shape], request, device):
    def linear(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.linear(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        linear,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(1, 32, 32), (2, 16, 16), (1, 1, 64)], ids=shape_str)
@pytest.mark.parametrize("dims", [[32, 1, 1], [1, 2, 2], [2, 3, 4], [1, 1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
def test_repeat(shape: Shape, dims: List[int], dtype, request, device):
    def repeat(
        in0: Operand, builder: TTNNBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.repeat(in0, dims=dims, unit_attrs=unit_attrs)

    compile_and_execute_ttnn(
        repeat,
        [shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 8, 1, 12, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("repeats", [1])
def test_repeat_interleave(
    shapes: List[Shape], repeats: int, dim: int, request, device
):
    def repeat_interleave(
        in0: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.repeat_interleave(
            in0, repeats=repeats, dim=dim, unit_attrs=unit_attrs
        )

    compile_and_execute_ttnn(
        repeat_interleave,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def concat(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    dim: int,
    builder: TTNNBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.concat([in0, in1, in2], dim=dim, unit_attrs=unit_attrs)


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (32, 128),
            (16, 128),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
def test_concat(shapes: List[Shape], dim: int, request, device):
    # Create a wrapper function that captures dim
    def concat_wrapper(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return concat(in0, in1, in2, dim, builder, unit_attrs)

    # Set the name for better test identification.
    concat_wrapper.__name__ = "concat"

    compile_and_execute_ttnn(
        concat_wrapper,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
