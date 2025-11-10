# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only

from builder.base.builder import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
    make_shard_shape,
    shard_wrap_factory,
)

pytestmark = pytest.mark.frontend("ttir")


def logical_not(
    in0: Operand,
    builder: TTIRBuilder,
    shape: Shape,
    dtype: torch.dtype,
    unit_attrs: Optional[List[str]] = None,
):
    randn_tensor = torch.randn(shape, dtype=torch.float32)
    input_tensor = randn_tensor.uniform_(-10.0, 10.0)
    input_tensor[torch.abs(input_tensor) < 4.0] = 0.0
    input_tensor = input_tensor.to(dtype)
    # Torch returns bool tensor but ttnn doesn't have bool type, convert to input dtype.
    golden_output_tensor = torch.logical_not(input_tensor).to(dtype)
    logical_not_0 = builder.logical_not(in0, unit_attrs=unit_attrs)
    builder.set_goldens({in0: input_tensor}, {logical_not_0: golden_output_tensor})
    return logical_not_0


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_logical_not(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def hoisted_logical_not_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return logical_not(in0, builder, shape, dtype, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        hoisted_logical_not_wrapper,
        [shape],
        [dtype],
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
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.clamp_scalar(
            in0, max_arg=max_arg, min_arg=min_arg, unit_attrs=unit_attrs
        )

    compile_and_execute_ttir(
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
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.clamp_tensor(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        clamp_tensor,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
    def dot_general(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.dot_general(
            in0,
            in1,
            batch_dims_lhs,
            contract_dims_lhs,
            batch_dims_rhs,
            contract_dims_rhs,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        dot_general,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def gt(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.gt(in0, in1, unit_attrs=unit_attrs)


def div(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    dividend_tensor = builder._get_golden_tensor(in0)
    divisor_tensor = builder._get_golden_tensor(in1)

    dividend_tensor = dividend_tensor.apply_shardwise(
        lambda shard: (
            shard.__setitem__(shard.abs() < 0.01, 0.03) or shard
            if torch.is_floating_point(shard)
            else shard
        )
    )

    divisor_tensor = divisor_tensor.apply_shardwise(
        lambda shard: (
            shard.__setitem__(shard.abs() < 0.01, -0.03) or shard
            if torch.is_floating_point(shard)
            else shard
        )
    )

    output_golden = torch.div(dividend_tensor, divisor_tensor)
    div0 = builder.div(in0, in1, unit_attrs=unit_attrs)
    builder.set_goldens_from_builder_tensor(
        {in0: dividend_tensor, in1: divisor_tensor}, {div0: output_golden}
    )
    return div0


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_div(shape: Shape, dtype: torch.dtype, target: str, request, device):
    compile_and_execute_ttir(
        div,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_div(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def hoisted_div_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        golden_input_tensor = torch.randn(shape, dtype=dtype)
        div0 = div(in0, in1, builder, unit_attrs=["ttir.should_hoist"])
        builder.set_goldens(
            {in0: golden_input_tensor, in1: golden_input_tensor},
            {div0: torch.div(golden_input_tensor, golden_input_tensor)},
        )
        return div0

    compile_and_execute_ttir(
        hoisted_div_wrapper,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shapes", [[(10, 64, 32), (32, 128), (128,)]], ids=shapes_list_str
)
def test_linear(shapes: List[Shape], request, device):
    def linear(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.linear(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        linear,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def matmul(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.matmul(in0, in1, unit_attrs=unit_attrs)


def sum(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sum(in0, unit_attrs=unit_attrs)


def mean(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.mean(in0, unit_attrs=unit_attrs)


def max(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.max(in0, unit_attrs=unit_attrs)


def min(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.min(in0, unit_attrs=unit_attrs)


@pytest.mark.xfail(reason="Fails Golden")
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dim_arg", [0])
@pytest.mark.parametrize("keep_dim", [False])
def test_prod(shape: Shape, dim_arg: int, keep_dim: bool, request, device):
    def prod(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.prod(in0, [dim_arg], keep_dim, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        prod,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def broadcast(
    in0: Operand,
    builder: TTIRBuilder,
    broadcast_dimensions: Optional[List[int]] = None,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.broadcast(
        in0, broadcast_dimensions=broadcast_dimensions, unit_attrs=unit_attrs
    )


@pytest.mark.parametrize("shape", [(1, 1, 32)], ids=shape_str)
@pytest.mark.parametrize("broadcast_dimensions", [[1, 16, 1]])
def test_broadcast(shape: Shape, broadcast_dimensions: List[int], request, device):
    # Create a wrapper function that captures broadcast_dimensions
    def broadcast_wrapper(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return broadcast(in0, builder, broadcast_dimensions, unit_attrs)

    # Set the name for better test identification
    broadcast_wrapper.__name__ = "broadcast"

    compile_and_execute_ttir(
        broadcast_wrapper,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 32, 3, 3),
            (1, 1, 1, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize(
    "input_dtypes",
    [
        [torch.float32, torch.float32, torch.float32],
        # skip quint8 for now. Issue: https://github.com/tenstorrent/tt-metal/issues/26568
        pytest.param(
            [
                TypeInfo(torch.quint8, scale=0.1, zero_point=128),
                TypeInfo(torch.qint8, scale=0.1, zero_point=0),
                torch.float32,
                torch.int8,
            ],
            marks=pytest.mark.skip(
                reason="Issue: https://github.com/tenstorrent/tt-metal/issues/26568"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([2, 1], [2, 1], [2, 1], 2)]
)
def test_conv2d(
    shapes: List[Shape],
    input_dtypes: List[Union[torch.dtype, TypeInfo]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
    device,
):
    def conv2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
            in0,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        conv2d,
        shapes,
        input_dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 32, 3, 3),
            (1, 1, 1, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("stride", [[2, 1]])
@pytest.mark.parametrize("dilation", [[2, 1]])
@pytest.mark.parametrize("padding", [[2, 1]])
@pytest.mark.parametrize("groups", [2])
def test_conv2d_consteval(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
    device,
):
    def conv2d_consteval(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
            in0,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        conv2d_consteval,
        shapes,
        argument_types_string="conv2d_consteval=input,parameter,parameter",
        test_base=request.node.name,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 64, 3, 3),
            (1, 1, 1, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("stride", [[2, 1]])
@pytest.mark.parametrize("dilation", [[2, 1]])
@pytest.mark.parametrize("padding", [[2, 1]])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_conv2d(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    target: str,
    request,
    device,
):
    """Test hoisted conv2d operation"""

    def hoisted_conv2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv2d(
            in0,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=["ttir.should_hoist"],
        )

    hoisted_conv2d.__name__ = "hoisted_conv2d"

    compile_and_execute_ttir(
        hoisted_conv2d,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (3, 8, 8, 256),
            (256, 256, 3, 3),
            (1, 1, 1, 256),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
@pytest.mark.parametrize(
    "stride,padding,output_padding,dilation,groups", [(1, 0, 0, 1, 1)]
)
def test_conv_transpose2d(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    groups: int,
    request,
    device,
):
    def conv_transpose2d(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.conv_transpose2d(
            in0,
            weight,
            bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        conv_transpose2d,
        shapes,
        dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode",
    [([2, 2], [2, 2], [1, 1], [0, 0, 0, 0], False)],
)
@pytest.mark.parametrize("shape", [(1, 128, 128, 32)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_max_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    request,
    device,
):
    def max_pool2d(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.max_pool2d(
            in0,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        max_pool2d,
        [shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode",
    [([2, 2], [2, 2], [1, 1], [0, 0, 0, 0], False)],
)
@pytest.mark.parametrize("shape", [(1, 128, 128, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_max_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    target: str,
    request,
    device,
):
    """Test hoisted max_pool2d operation"""

    def hoisted_max_pool2d(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.max_pool2d(
            in0,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            unit_attrs=["ttir.should_hoist"],
        )

    hoisted_max_pool2d.__name__ = "hoisted_max_pool2d"

    compile_and_execute_ttir(
        hoisted_max_pool2d,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "kernel,stride,dilation,padding,ceil_mode,count_include_pad",
    [
        ([2, 2], [2, 2], [1, 1], [1, 1, 1, 1], False, True),
        (
            [2, 2],
            [1, 1],
            [1, 1],
            [1, 1, 1, 1],
            True,
            False,
        ),  # This test will produce a different output if count_include_pad is True for spatial dims (31, 31)
    ],
)
@pytest.mark.parametrize("shape", [(1, 31, 31, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_avg_pool2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel: List[int],
    stride: List[int],
    dilation: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    request,
    device,
):
    def avg_pool2d(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.avg_pool2d(
            in0,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        avg_pool2d,
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
            (1, 64, 32, 32),  # input tensor: (N, C, H, W)
            (64,),  # scale (gamma)
            (64,),  # offset (beta)
            (64,),  # mean
            (64,),  # variance
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 5])
@pytest.mark.parametrize("dimension", [1])  # channel dimension
@pytest.mark.parametrize("epsilon", [1e-5])
def test_batch_norm(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    dimension: int,
    epsilon: float,
    request,
    device,
):
    def batch_norm(
        in0: Operand,
        scale: Operand,
        offset: Operand,
        mean: Operand,
        variance: Operand,
        builder,
        unit_attrs: Optional[List[str]] = None,
    ):

        return builder.batch_norm(
            in0,
            scale,
            offset,
            mean,
            variance,
            epsilon=epsilon,
            dimension=dimension,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        batch_norm,
        shapes,
        dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(32, 64)], ids=shape_str)
@pytest.mark.parametrize("dim,begin,end,step", [(0, 0, 3, 1)])
def test_index(
    shape: Shape, dim: int, begin: int, end: int, step: int, request, device
):
    def index(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.index(
            in0, dim=dim, begin=begin, end=end, step=step, unit_attrs=unit_attrs
        )

    compile_and_execute_ttir(
        index,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(4, 4)], ids=shape_str)
@pytest.mark.parametrize("dim,begin,length,stride", [(1, 2, 2, 2)])
def test_select(
    shape: Shape, dim: int, begin: int, length: int, stride: int, request, device
):
    def select(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.select(
            in0,
            dim=dim,
            begin=begin,
            length=length,
            stride=stride,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        select,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# TODO (ctod): These three nullary tensor creation ops can probably be combined in some way.
@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int32], ids=["bf16", "f32", "i32"]
)
def test_zeros(shape: Shape, dtype: torch.dtype, request, device):
    def zeros(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.zeros(shape, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        zeros,
        inputs_shapes=[],
        inputs_types=[],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
def test_ones(shape: Shape, request, device):
    def ones(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.ones(shape, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        ones,
        inputs_shapes=[],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "tensor",
    [
        torch.tensor(1, dtype=torch.uint8),
        torch.tensor([1, 2, 3, 4], dtype=torch.uint16),
        torch.tensor([1, 2, 3, 4], dtype=torch.uint32),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.bfloat16),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    ],
    ids=[
        "scalar_int-uint8",
        "1d_int-uint16",
        "1d_int-uint32",
        "2d_int-int32",
        "1d_float-bf16",
        "1d_float-f32",
        "torch_float_tensor-float32",
    ],
)
def test_constant(tensor, request, device):
    def constant(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.constant(tensor, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        constant,
        [],
        [],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_basic(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    """Basic test demonstrating callable initialization with torch.zeros and torch.ones"""

    def test_with_basic_callables(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_goldens({in0: torch.zeros, in1: torch.ones})
        result = builder.add(in0, in1, unit_attrs=unit_attrs)
        return result

    compile_and_execute_ttir(
        test_with_basic_callables,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_zeros(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def test_with_zeros_init(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        builder.set_goldens({in0: torch.zeros})
        zeros_result = builder.neg(in0, unit_attrs=unit_attrs)
        return zeros_result

    compile_and_execute_ttir(
        test_with_zeros_init,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_ones(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def test_with_ones_init(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        builder.set_goldens({in0: torch.ones})
        ones_result = builder.neg(in0, unit_attrs=unit_attrs)
        return ones_result

    compile_and_execute_ttir(
        test_with_ones_init,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_eye(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def test_with_eye_init(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        def eye_init(s):
            if len(s) == 2 and s[0] == s[1]:
                return torch.eye(s[0])
            elif len(s) == 2:
                return torch.eye(s[0], s[1])
            else:
                raise ValueError(f"torch.eye only supports 2D shapes, got {s}")

        builder.set_goldens({in0: eye_init})
        eye_result = builder.abs(in0, unit_attrs=unit_attrs)
        return eye_result

    compile_and_execute_ttir(
        test_with_eye_init,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_mixed(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def test_with_mixed_init(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_goldens({in0: torch.zeros, in1: torch.ones})
        add_result = builder.add(in0, in1, unit_attrs=unit_attrs)
        return add_result

    compile_and_execute_ttir(
        test_with_mixed_init,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_custom_lambda(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def test_with_custom_lambda(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        custom_init = lambda s: torch.full(s, 2.0)
        builder.set_goldens({in0: custom_init})
        result = builder.multiply(in0, in0, unit_attrs=unit_attrs)
        return result

    compile_and_execute_ttir(
        test_with_custom_lambda,
        [shape],
        [dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_callable_initialization_error_handling(shape: Shape, dtype: torch.dtype):
    """Test error handling for invalid callable initialization functions"""

    def test_with_invalid_callable(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        invalid_init = lambda s: "not a tensor"

        result = builder.neg(in0, unit_attrs=unit_attrs)
        with pytest.raises((TypeError, RuntimeError)):
            builder.set_goldens({in0: invalid_init})
        return result

    def test_with_failing_callable(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        failing_init = lambda s: torch.zeros(s) / 0  # Division by zero

        result = builder.neg(in0, unit_attrs=unit_attrs)
        with pytest.raises(RuntimeError):
            builder.set_goldens({in0: failing_init})
        return result


@pytest.mark.parametrize("shapes", [[(128, 128)]], ids=shapes_list_str)
@pytest.mark.parametrize("dim_arg", [[1]])
def test_argmax(shapes, dim_arg, request, device):
    def argmax(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.argmax(in0, dim_arg, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        argmax,
        inputs_shapes=shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(256, 128, 2, 2)], ids=shape_str)
@pytest.mark.parametrize("dims", [[2, 3]])
def test_reverse(shape: Shape, dims: List[int], request, device):
    def reverse(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reverse(in0, dims=dims, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        reverse,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip(reason="See issue #3685")
@pytest.mark.parametrize("shape", [(4, 4)])
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_and(shape: Shape, dim_args: List[int], request, device):
    def reduce_and(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reduce_and(in0, dim_args=dim_args, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        reduce_and,
        [shape],
        [torch.int32],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def reduce_or(
    in0: Operand,
    builder: TTIRBuilder,
    dim_args: List[int],
    keep_dim: bool = False,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_or(
        in0, dim_args=dim_args, keep_dim=keep_dim, unit_attrs=unit_attrs
    )


@pytest.mark.xfail(reason="only floats are supported in runtime. See issue #1775")
@pytest.mark.parametrize("shape", [(4, 4)], ids=shape_str)
@pytest.mark.parametrize("dim_args", [[0, 1]])
def test_reduce_or(shape: Shape, dim_args: List[int], request, device):
    def reduce_or_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return reduce_or(in0, builder, dim_args=dim_args, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        reduce_or_wrapper,
        [shape],
        [torch.int32],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shapes", [[(10, 64, 32, 3), (10, 128, 128, 3)]], ids=shapes_list_str
)
@pytest.mark.parametrize("scale_factor", [[2, 4]])
def test_upsample2d(shapes: List[Shape], scale_factor: List[int], request, device):
    def upsample2d(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.upsample2d(
            in0,
            in1,
            scale_factor=scale_factor,
            unit_attrs=unit_attrs,
        )

    compile_and_execute_ttir(
        upsample2d,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape,start,end,step,dim", [((5,), 0, 5, 1, 0)])
def test_arange(
    shape: Shape, start: int, end: int, step: int, dim: int, request, device
):
    def arange(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.arange(in0, start, end, step, dim, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        arange,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(4, 4, 128, 128)]], ids=shapes_list_str)
@pytest.mark.parametrize("dim", [1])
def test_cumsum(shapes: List[Shape], dim: int, request, device):
    def cumsum(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.cumsum(in0, dim=dim, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        cumsum,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def prod(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.prod(in0, [1], False, unit_attrs=unit_attrs)


@pytest.mark.xfail(reason="Fails Golden")
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 64, 512), (1, 32, 3, 512)]], ids=shapes_list_str
)
def test_fill_cache(shapes: List[Shape], request, device):
    def fill_cache(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.fill_cache(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        fill_cache,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def softmax(
    in0: Operand,
    builder: TTIRBuilder,
    dimension: int = -1,
    numeric_stable: bool = False,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.softmax(
        in0, dimension=dimension, numeric_stable=numeric_stable, unit_attrs=unit_attrs
    )


@pytest.mark.parametrize("shape", [(512, 1024)], ids=shape_str)
@pytest.mark.parametrize("dimension", [-1])
@pytest.mark.parametrize("numeric_stable", [False, True])
def test_softmax(shape: Shape, dimension: int, numeric_stable: bool, request, device):
    # Create a wrapper function that captures dimension
    def softmax_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return softmax(in0, builder, dimension, numeric_stable, unit_attrs)

    # Set the name for better test identification
    softmax_wrapper.__name__ = "softmax"

    compile_and_execute_ttir(
        softmax_wrapper,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.xfail(reason="run error")
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 64, 512), (1, 32, 1, 512), (1,)]], ids=shapes_list_str
)
@pytest.mark.parametrize("dtypes", [[torch.float32, torch.float32, torch.int32]])
def test_update_cache(shapes: List[Shape], dtypes: List[torch.dtype], request, device):
    def update_cache(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.update_cache(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        update_cache,
        shapes,
        inputs_types=dtypes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def embedding(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.embedding(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.qint32,
        pytest.param(
            torch.qint8,
            marks=pytest.mark.skip(
                reason="qint8 quantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
    ids=["qint32", "qint8"],
)
def test_quantize(
    shape: Shape,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
    device,
):
    def quantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.quantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        quantize,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "input_dtype",
    [
        TypeInfo(torch.qint32, 0.1, 0),
        pytest.param(
            TypeInfo(torch.qint8, 0.1, 0),
            marks=pytest.mark.skip(
                reason="qint8 dequantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
    ids=["qint32", "qint8"],
)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_dequantize(
    shape: Shape,
    input_dtype: TypeInfo,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
    device,
):
    def dequantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.dequantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        dequantize,
        [shape],
        inputs_types=[input_dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "input_dtype",
    [
        TypeInfo(torch.qint32, 0.1, 0),
        pytest.param(
            TypeInfo(torch.qint8, 0.1, 0),
            marks=pytest.mark.skip(
                reason="qint8 requantize not supported. issue https://github.com/tenstorrent/tt-metal/issues/26414"
            ),
        ),
    ],
)
@pytest.mark.parametrize("scale", [0.1])
@pytest.mark.parametrize("zero_point", [0])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.qint32,
        pytest.param(
            torch.qint8, marks=pytest.mark.skip(reason="qint8 quantize not supported")
        ),
    ],
    ids=["qint32", "qint8"],
)
def test_requantize(
    shape: Shape,
    input_dtype: TypeInfo,
    scale: float,
    zero_point: int,
    dtype: torch.dtype,
    request,
    device,
):
    def requantize(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.requantize(in0, scale, zero_point, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        requantize,
        [shape],
        inputs_types=[input_dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Create hoisted versions of operations by currying the unit_attrs parameter
def create_hoisted_single_operand_op(op_func, name):
    """Create a hoisted version of a unary operation by adding the should_hoist unit attribute"""

    def hoisted_op(in0, builder, **kwargs):
        # For unary ops
        return op_func(in0, builder, unit_attrs=["ttir.should_hoist"], **kwargs)

    # Set the name for better test identification
    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_permute_op(op_func, name):
    """Create a hoisted version of the permute operation that calculates appropriate permutation dimensions"""

    def hoisted_op(in0, in1, builder, **kwargs):
        # Calculate appropriate permutation based on input dimensions
        input_shape = builder.get_shape(in0)
        ndims = len(input_shape)

        # Create a simple permutation that reverses the dimensions
        # This is guaranteed to be valid for any tensor
        permutation = list(range(ndims))
        permutation.reverse()

        return op_func(
            in0, in1, builder, permutation, unit_attrs=["ttir.should_hoist"], **kwargs
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_softmax_op(op_func, name):
    """Create a hoisted version of the softmax operation"""

    def hoisted_op(in0, builder, **kwargs):
        # Default dimension for the hoisted version (last dimension)
        default_dimension = -1
        return op_func(
            in0,
            builder,
            dimension=default_dimension,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


def create_hoisted_concat_op(op_func, name):
    """Create a hoisted version of the concat operation"""

    def hoisted_op(in0, in1, in2, builder, **kwargs):
        # Default dimension for the hoisted version (dimension 0)
        default_dim = 0
        return op_func(
            in0,
            in1,
            in2,
            default_dim,
            builder,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create a function for hoisted where operation
def create_hoisted_where_op(op_func, name):
    """Create a hoisted version of the where operation"""

    def hoisted_op(condition, x, y, builder, **kwargs):
        return op_func(
            condition, x, y, builder, unit_attrs=["ttir.should_hoist"], **kwargs
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create a function for hoisted slice operation
def create_hoisted_slice_op(op_func, name):
    """Create a hoisted version of the slice operation"""

    def hoisted_op(in0, builder, **kwargs):
        # Default slice parameters
        begins = DenseI32ArrayAttr.get([0, 0])
        ends = DenseI32ArrayAttr.get([10, 10])
        steps = DenseI32ArrayAttr.get([1, 1])
        return op_func(
            in0,
            begins,
            ends,
            steps,
            builder,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create a function for hoisted reduce operations
def create_hoisted_reduce_op(op_func, name):
    """Create a hoisted version of a reduce operation that requires dimension arguments"""

    def hoisted_op(in0, builder, **kwargs):
        # Default dimension arguments for the hoisted version
        default_dim_args = [0]  # Use first dimension as default
        return op_func(
            in0,
            builder,
            dim_args=default_dim_args,
            unit_attrs=["ttir.should_hoist"],
            **kwargs,
        )

    hoisted_op.__name__ = f"hoisted_{name}"
    return hoisted_op


# Create hoisted versions of all hoistable operations with proper names
hoisted_single_operand_ops = [
    create_hoisted_single_operand_op(sum, "sum"),
    pytest.param(
        create_hoisted_single_operand_op(softmax, "softmax"),
        marks=pytest.mark.xfail(
            reason="Softmax does not lower to loops properly https://github.com/tenstorrent/tt-mlir/issues/3232"
        ),
    ),
    create_hoisted_single_operand_op(mean, "mean"),
]


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("test_fn", hoisted_single_operand_ops)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_single_operand_ops(
    test_fn: Callable,
    shape: Shape,
    request,
    target: str,
    device,
    dtype: torch.dtype = torch.float32,
):
    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttir(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=f"{request.node.name}",
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Test hoisted permute separately because it requires unique input shapes.
@x86_only
@pytest.mark.parametrize(
    "shapes,permutation",
    [
        # [(input_shape, output_shape), permutation]
        ([(2, 3, 4), (4, 2, 3)], [2, 0, 1]),
        ([(128, 128), (128, 128)], [0, 1]),
        ([(128, 64, 32), (32, 128, 64)], [2, 0, 1]),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.xfail(reason="Fails Golden")
def test_hoisted_permute(shapes, permutation, request, target: str, device):
    def permute_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return permute(in0, in1, builder, permutation, unit_attrs=["ttir.should_hoist"])

    permute_wrapper.__name__ = "hoisted_permute"

    compile_and_execute_ttir(
        permute_wrapper,
        shapes,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Test hoisted max separately because it requires more complex parameters combination.
@x86_only
@pytest.mark.parametrize("dim_arg", [None, 0, 1])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 10), (10, 1), (64, 32), (128, 64), (128, 128)], ids=shape_str
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_max(shape, dim_arg, keep_dim, request, target: str, device):
    if keep_dim is False:
        pytest.skip(
            "Known mismatch: TTIR keep_dim=False rank change unsupported by TOSA reduce op; "
            "see issue #5061 - https://github.com/tenstorrent/tt-mlir/issues/5061"
        )

    def max(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
        return builder.max(
            in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=["ttir.should_hoist"]
        )

    max.__name__ = "hoisted_max"
    compile_and_execute_ttir(
        max,
        [shape],
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shape,begins,ends,step",
    [
        ((64, 64), [0, 0], [32, 32], None),
        ((128, 128), [10, 20], [50, 60], [1, 1]),
        ((32, 64, 64), [5, 10, 15], [25, 50, 55], [2, 2, 1]),
    ],
    ids=["basic_slice", "explicit_step", "3d_slice"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_slice(
    shape: Shape,
    begins: List[int],
    ends: List[int],
    step: List[int],
    target: str,
    request,
    device,
):
    def slice_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        # Now use the slice operation with the CPU hoisting attribute
        return builder.slice(in0, begins, ends, step, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        slice_wrapper,
        [shape],
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Add test for hoisted where operation
@x86_only
@pytest.mark.parametrize(
    "shapes", [[(64, 64), (64, 64), (64, 64)]], ids=shapes_list_str
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_where(shapes, request, target: str, device):
    def where_wrapper(condition: Operand, x: Operand, y: Operand, builder: TTIRBuilder):
        return builder.where(condition, x, y, unit_attrs=["ttir.should_hoist"])

    where_wrapper.__name__ = "hoisted_where"

    compile_and_execute_ttir(
        where_wrapper,
        shapes,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "input_shape,output_shape",
    [
        # (input_shape, output_shape)
        ((2, 3, 4), (24,)),
        ((128, 128), (16384,)),
        ((128, 64, 32), (128, 2048)),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_reshape(input_shape, output_shape, request, target: str, device):
    def reshape_wrapper(in0: Operand, builder: TTIRBuilder):
        return builder.reshape(in0, output_shape, unit_attrs=["ttir.should_hoist"])

    reshape_wrapper.__name__ = "hoisted_reshape"

    compile_and_execute_ttir(
        reshape_wrapper,
        [input_shape],
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "input_shape,dims",
    [
        # (input_shape, permutation)
        ((2, 3, 4), [2, 1]),
        ((128, 128), [1, 0]),
        ((128, 64, 32), [0, 2]),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_transpose(input_shape, dims, request, target: str, device):
    def transpose_wrapper(in0: Operand, builder: TTIRBuilder):
        # For 2D tensors with permutation [1, 0], swap dimensions 0 and 1
        # For 3D tensors with permutation [2, 1, 0], swap dimensions 0 and 2
        dim0 = dims[0]
        dim1 = dims[1]
        return builder.transpose(
            in0, dim0=dim0, dim1=dim1, unit_attrs=["ttir.should_hoist"]
        )

    transpose_wrapper.__name__ = "hoisted_transpose"

    compile_and_execute_ttir(
        transpose_wrapper,
        [input_shape],
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize("shape", [(1, 128, 128)], ids=shape_str)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_squeeze(shape: Shape, dim: int, target: str, request, device):
    """Test hoisted squeeze operation with appropriate shape that has a dimension of size 1"""

    def hoisted_squeeze(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.squeeze(in0, dim=dim, unit_attrs=["ttir.should_hoist"])

    hoisted_squeeze.__name__ = "hoisted_squeeze"

    compile_and_execute_ttir(
        hoisted_squeeze,
        [shape],
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


reduction_ops = [
    sum | Marks(pytest.mark.skip_config(["ttmetal"])),
    mean | Marks(pytest.mark.skip_config(["ttmetal"])),
    max
    | Marks(
        pytest.mark.skip_config(["ttnn"]),
        pytest.mark.skip_config(["ttmetal"]),
    ),
    min
    | Marks(
        pytest.mark.skip_config(["ttnn"]),
        pytest.mark.skip_config(["ttmetal"]),
    ),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitc", "emitpy"])
@pytest.mark.parametrize("test_fn", reduction_ops)
def test_reduction_ops(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    pipeline_options = []
    compile_and_execute_ttir(
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


unary_ops_int32 = [
    pytest.param(
        sum,
        marks=pytest.mark.skip(
            reason="Sum does not support int32 input. Issue: https://github.com/tenstorrent/tt-metal/issues/26724"
        ),
    ),
    pytest.param(
        max,
        marks=pytest.mark.skip(
            reason="Max does not support int32 input. Issue: https://github.com/tenstorrent/tt-metal/issues/26726"
        ),
    ),
    pytest.param(
        min,
        marks=pytest.mark.skip(
            reason="Min does not support int32 input. Issue: https://github.com/tenstorrent/tt-metal/issues/26726"
        ),
    ),
]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
# TODO (anuragsingh): Add tt-metal and emitc tests. Link to issue: https://github.com/tenstorrent/tt-mlir/issues/4444
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("test_fn", unary_ops_int32)
def test_unary_ops_int32(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    pipeline_options = []
    compile_and_execute_ttir(
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


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
@pytest.mark.parametrize(
    "test_fn",
    [
        matmul | Marks(pytest.mark.skip_config(["ttmetal"])),
    ],
)
def test_matmul(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    # NOTE: this function is _only_ for binary ops that take the same shape arguments
    pipeline_options = []
    compile_and_execute_ttir(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize(
    "test_fn,inputs_shapes,inputs_dtypes",
    [
        pytest.param(
            embedding,
            [(33, 32), (512, 128)],
            [torch.float32] * 2,
            marks=[pytest.mark.skip_config(["ttmetal"])],
        ),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_unique_ops(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_dtypes: List[torch.dtype],
    target: str,
    request,
    device,
):
    compile_and_execute_ttir(
        test_fn,
        inputs_shapes=inputs_shapes,
        inputs_types=inputs_dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize("shape", [(4, 4)], ids=shape_str)
@pytest.mark.parametrize("dim_args", [[0]])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.xfail(reason="Issue #3883")
def test_hoisted_reduce_or(
    shape: Shape, dim_args: List[int], target: str, request, device
):
    """Test the hoisted reduce_or operation with proper dimensions and keep_dim parameter"""

    def hoisted_reduce_or_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return reduce_or(
            in0, builder, dim_args, keep_dim=True, unit_attrs=["ttir.should_hoist"]
        )

    compile_and_execute_ttir(
        hoisted_reduce_or_wrapper,
        inputs_shapes=[shape],
        inputs_types=[torch.float32],
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shape,broadcast_dims",
    [
        # [input_shape, broadcast_dimensions]
        ((1, 1, 32), [1, 16, 1]),
        ((128, 1), [1, 64]),
        ((1, 128), [64, 1]),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_broadcast(shape, broadcast_dims, request, target: str, device):
    """Test broadcast operation with CPU hoisting enabled using the 'hoisted_' naming convention"""

    def broadcast_wrapper(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return broadcast(in0, builder, broadcast_dims, unit_attrs=["ttir.should_hoist"])

    broadcast_wrapper.__name__ = "hoisted_broadcast"

    compile_and_execute_ttir(
        broadcast_wrapper,
        inputs_shapes=[shape],
        test_base=f"{request.node.name}",
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


def gather(
    in0: Operand,
    builder: TTIRBuilder,
    indices_shape: Shape,
    start_index_map: List[int],
    offset_dims: List[int],
    slice_sizes: List[int],
    indices_dtype: torch.dtype,
    unit_attrs: Optional[List[str]] = None,
):
    # For now, just create zero indices - this tests the basic gather functionality.
    # In a real test, you'd want to create varied indices to test different gather patterns.
    indices = builder.zeros(indices_shape, indices_dtype)

    # Set collapsed_slice_dims to be the same as start_index_map
    # This is what the GatherToEmbeddingConversionPattern expects.
    collapsed_slice_dims = start_index_map

    # Set remaining parameters to empty lists for simplicity.
    operand_batching_dims = []
    start_indices_batching_dims = []

    # Set index_vector_dim correctly based on the use case.
    if len(indices_shape) == 1 and len(start_index_map) == 1:
        # Single indices case - index vector dim is implicit.
        index_vector_dim = len(indices_shape)  # = 1
    else:
        # Multi-dimensional indices - last dimension contains index vectors.
        index_vector_dim = len(indices_shape) - 1

    return builder.gather(
        in0,
        indices,
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        operand_batching_dims=operand_batching_dims,
        start_indices_batching_dims=start_indices_batching_dims,
        start_index_map=start_index_map,
        index_vector_dim=index_vector_dim,
        slice_sizes=slice_sizes,
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize(
    "input_shape,input_dtype,indices_shape,start_index_map,offset_dims,slice_sizes",
    [
        # Simple 1D indices - f32.
        ((100, 50), torch.float32, (10,), [0], [1], [1, 50]),
        pytest.param(
            (8, 16, 32),
            torch.float32,
            (4, 2, 2),
            [0, 2],
            [1],
            # Complex indices - f32.
            [1, 16, 1],
        ),
    ],
    ids=[
        "simple_1d-f32",
        "complex_indices-f32",
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_gather(
    input_shape: Shape,
    input_dtype: torch.dtype,
    indices_shape: Shape,
    start_index_map: List[int],
    offset_dims: List[int],
    slice_sizes: List[int],
    target: str,
    request,
    device,
):
    def gather_wrapper(in0: Operand, builder: TTIRBuilder):
        return gather(
            in0,
            builder,
            indices_shape,
            start_index_map,
            offset_dims,
            slice_sizes,
            input_dtype,
        )

    compile_and_execute_ttir(
        gather_wrapper,
        [input_shape],
        [input_dtype],
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "input_shape,input_dtype,indices_shape,start_index_map,offset_dims,slice_sizes",
    [
        ((100, 50), torch.float32, (10,), [0], [1], [1, 50]),  # Simple 1D indices
        pytest.param(
            (8, 16, 32),
            torch.float32,
            (4, 2, 2),
            [0, 2],
            [1],
            [1, 16, 1],
        ),  # Complex indices
    ],
    ids=["simple_1d", "complex_indices"],
)
@pytest.mark.parametrize(
    "target",
    ["ttnn", "ttmetal" | Marks(pytest.mark.xfail(reason="Unhoisted ttir.zeros"))],
)
def test_hoisted_gather(
    input_shape: Shape,
    input_dtype: torch.dtype,
    indices_shape: Shape,
    start_index_map: List[int],
    offset_dims: List[int],
    slice_sizes: List[int],
    target: str,
    request,
    device,
):
    def gather_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return gather(
            in0,
            builder,
            indices_shape,
            start_index_map,
            offset_dims,
            slice_sizes,
            input_dtype,
            unit_attrs=["ttir.should_hoist"],
        )

    compile_and_execute_ttir(
        gather_wrapper,
        [input_shape],
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.skip(reason="https://github.com/tenstorrent/tt-mlir/issues/5315")
@x86_only
@pytest.mark.parametrize(
    "shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
    [
        # Standard matrix multiplication: [M, K] x [K, N] -> [M, N]
        ([(10, 20), (20, 30), (10, 30)], [], [1], [], [0]),
        # Batched matrix multiplication: [B, M, K] x [B, K, N] -> [B, M, N]
        ([(5, 10, 20), (5, 20, 30), (5, 10, 30)], [0], [2], [0], [1]),
        # 3D tensor @ 2D tensor: [B, M, K] x [K, N] -> [B, M, N]
        ([(5, 10, 20), (20, 30), (5, 10, 30)], [], [2], [], [0]),
    ],
    ids=["standard_matmul", "batched_matmul", "3d_tensor_2d_tensor"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_dot_general(
    shapes: List[Shape],
    batch_dims_lhs: List[int],
    contract_dims_lhs: List[int],
    batch_dims_rhs: List[int],
    contract_dims_rhs: List[int],
    target: str,
    request,
    device,
):
    def dot_general_wrapper(
        in0: Operand,
        in1: Operand,
        out0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.dot_general(
            in0,
            in1,
            out0,
            batch_dims_lhs,
            contract_dims_lhs,
            batch_dims_rhs,
            contract_dims_rhs,
            unit_attrs=["ttir.should_hoist"],
        )

    compile_and_execute_ttir(
        dot_general_wrapper,
        shapes,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        [(10, 20), (20, 30)],
        [(5, 10, 20), (5, 20, 30)],
    ],
    ids=["standard_2D_matmul", "3D_batched_matmul"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_matmul(
    shapes: List[Shape], dtype: torch.dtype, target: str, request, device
):
    def hoisted_matmul_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return matmul(in0, in1, builder, unit_attrs=["ttir.should_hoist"])

    hoisted_matmul_wrapper.__name__ = "hoisted_matmul"

    compile_and_execute_ttir(
        hoisted_matmul_wrapper,
        shapes,
        [dtype] * len(shapes),
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((32, 128), [128]),
        ((2, 4, 64), [64]),
    ],
)
@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_rms_norm(
    shape: Shape,
    normalized_shape: List[int],
    has_weight: bool,
    has_bias: bool,
    target: str,
    request,
    device,
):
    def rms_norm(*inputs, unit_attrs: Optional[List[str]] = None):

        builder = inputs[-1]
        # Extract inputs based on test configuration
        in0 = inputs[0]
        weight = None
        bias = None

        if has_weight and len(inputs) > 1:
            weight = inputs[1]
        if has_bias:
            if has_weight and len(inputs) > 2:
                bias = inputs[2]
            elif not has_weight and len(inputs) > 1:
                bias = inputs[1]

        return builder.rms_norm(
            in0,
            normalized_shape=normalized_shape,
            weight=weight,
            bias=bias,
            unit_attrs=unit_attrs,
        )

    # Determine input shapes
    shapes = [shape]
    if has_weight:
        shapes.append(tuple(normalized_shape))
    if has_bias:
        shapes.append(tuple(normalized_shape))

    compile_and_execute_ttir(
        rms_norm,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@pytest.mark.parametrize(
    "input_rank, shard_dims",
    [
        (5, (1, 4)),
        (5, (4, 1)),
        (5, (2, 4)),
        (5, (1, 4)),
        (5, (-1, 3)),
        (5, (4, -1)),
        (5, (-1, 4)),
        (5, (-1, 0)),
        (4, (1, 3)),
        (4, (3, 1)),
        (4, (2, 3)),
        (4, (3, 2)),
        (4, (0, 2)),
        (4, (1, 0)),
        (4, (-1, 3)),
        (4, (3, -1)),
        (4, (-1, 1)),
        (4, (1, -1)),
        (3, (1, 2)),
        (3, (2, 1)),
        (3, (0, 1)),
        (3, (1, 0)),
        (3, (-1, 2)),
        (3, (2, -1)),
        (3, (-1, 1)),
        (3, (0, -1)),
        (2, (0, 1)),
        (2, (1, 0)),
        (2, (-1, 1)),
        (2, (1, -1)),
        (2, (-1, 0)),
        (2, (0, -1)),
    ],
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (4, 2), (1, 8), (8, 1), (1, 2), (2, 1)], ids=shape_str
)
def test_mesh_shard_devices(
    input_rank: int,
    shard_dims: Tuple[int, int],
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    shard_shape = make_shard_shape(input_rank, shard_dims, mesh_shape)
    if all(x == 1 for x in shard_shape):
        pytest.skip("sharding is meaningless, skipping test.")
    input_shape = [n_shards for idx, n_shards in enumerate(shard_shape)]

    def mesh_shard_devices(in0: Operand, builder: TTIRBuilder):
        mesh_shard_in0 = builder.mesh_shard(
            in0,
            shard_direction="#ttcore.shard_direction<full_to_shard>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )
        neg_output = builder.neg(mesh_shard_in0)
        return builder.mesh_shard(
            neg_output,
            shard_direction="#ttcore.shard_direction<shard_to_full>",
            shard_type="#ttcore.shard_type<devices>",
            shard_shape=shard_shape,
            shard_dims=shard_dims,
        )

    compile_and_execute_ttir(
        mesh_shard_devices,
        [input_shape],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 32, 32, 32),
        (1, 32, 32, 1),
        (32, 32, 1, 1),
        (1, 32, 32),
        (32, 32),
        (32, 40),
        (40, 32),
        pytest.param((1, 1, 32, 32, 32), marks=pytest.mark.xfail(reason="run error")),
        pytest.param(
            (1, 1, 1, 1, 1, 1, 32, 32, 32), marks=pytest.mark.xfail(reason="run error")
        ),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
@pytest.mark.parametrize("all_gather_dim", range(4))
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_gather(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    all_gather_dim: int,
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    if all_gather_dim >= len(test_shape):
        pytest.skip("all_gather_dim is out of range")
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("all_gather across 1 device is meaningless")

    def all_gather(mesh_shard_in: Operand, builder: TTIRBuilder):
        return builder.all_gather(
            mesh_shard_in,
            all_gather_dim=all_gather_dim,
            cluster_axis=cluster_axis,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, all_gather)

    compile_and_execute_ttir(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        pytest.param((1, 1, 1, 256, 256), marks=pytest.mark.xfail(reason="run error")),
        (1, 1, 256, 256),
        (1, 1, 256, 257),
        (1, 1, 256, 255),
        (1, 256, 256, 1),
        (256, 256, 1, 1),
        (1, 1, 32, 64),
        (1, 64, 64),
        (64, 64),
        (64, 65),
        (32, 64),
        pytest.param(
            (33, 65), marks=pytest.mark.xfail(reason="run error")
        ),  # all_gather + local reduce case
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_reduce(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("CCL across 1 device is meaningless")

    # test 'sum' only for now. Other reduce types are not supported yet.
    def all_reduce(mesh_shard_in: Operand, builder: TTIRBuilder):
        return builder.all_reduce(
            mesh_shard_in,
            reduce_type="#ttcore.reduce_type<sum>",
            cluster_axis=cluster_axis,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, all_reduce)

    compile_and_execute_ttir(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 256, 256),
        (1, 1, 256, 257),
        (1, 1, 256, 255),
        (1, 256, 256, 1),
        (256, 256, 1, 1),
        (1, 1, 32, 64),
        (1, 128, 128),
        (128, 128),
        (128, 129),
        (64, 128),
        (64, 24),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape", [(2, 4), (1, 8), (1, 2), (1, 32), (8, 4)], ids=shape_str
)
@pytest.mark.parametrize("scatter_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("cluster_axis", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_reduce_scatter(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    scatter_dim: int,
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    if mesh_shape[cluster_axis] == 1:
        pytest.skip("CCL across 1 device is meaningless")
    if scatter_dim >= len(test_shape):
        pytest.skip("scatter_dim is out of range")
    if test_shape[scatter_dim] % mesh_shape[cluster_axis] != 0:
        pytest.skip("scatter_dim is not divisible by mesh_shape[cluster_axis]")

    # test 'sum' only for now. Other reduce types are not supported yet.
    def reduce_scatter(mesh_shard_in: Operand, builder: TTIRBuilder):
        return builder.reduce_scatter(
            mesh_shard_in,
            reduce_type="#ttcore.reduce_type<sum>",
            scatter_dim=scatter_dim,
            cluster_axis=cluster_axis,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, reduce_scatter)

    compile_and_execute_ttir(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 32, 64),
        (1, 32, 64),
        (32, 64),
        (30, 60),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape, source_target_pairs",
    [
        pytest.param(
            (1, 2), [(0, 1)], marks=pytest.mark.xfail(reason="Fails Golden")
        ),  # https://github.com/tenstorrent/tt-mlir/issues/4323
        ((1, 2), [(0, 1), (1, 0)]),
        ((2, 4), [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ((2, 4), [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]),
        ((2, 4), [(0, 4), (4, 0), (1, 5), (5, 1), (2, 6), (6, 2), (3, 7), (7, 3)]),
        ((2, 4), [(0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)]),
        ((2, 4), [(0, 2), (1, 3), (4, 6), (5, 7), (2, 0), (3, 1), (6, 4), (7, 5)]),
        ((2, 4), [(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)]),
        pytest.param(
            (2, 4),
            [(0, 1), (2, 3), (4, 5), (6, 7)],
            marks=pytest.mark.xfail(
                reason="https://github.com/tenstorrent/tt-mlir/issues/4323"
            ),
        ),
        ((1, 8), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]),
        ((1, 32), [(i, (i + 1) % 32) for i in range(32)]),
        (
            (8, 4),
            # fmt: off
            # rotate right within each cluster along axis 1
            [
                (0, 1), (1, 2), (2, 3), (3, 0), # cluster #0
                (4, 5), (5, 6), (6, 7),( 7, 4), # cluster #1
                (8, 9), (9, 10), (10, 11), (11, 8), # cluster #2
                (12, 13), (13, 14), (14, 15), (15, 12), # cluster #3
                (16, 17), (17, 18), (18, 19), (19, 16), # cluster #4
                (20, 21), (21, 22), (22, 23), (23, 20), # cluster #5
                (24, 25), (25, 26), (26, 27), (27, 24), # cluster #6
                (28, 29), (29, 30), (30, 31), (31, 28), # cluster #7
            ],
            # fmt: on
        ),
        (
            (8, 4),
            # fmt: off
            # rotate right within each cluster along axis 0
            [
                (0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 0), # cluster #0
                (1, 5), (5, 9), (9, 13), (13, 17), (17, 21), (21, 25), (25, 29), (29, 1), # cluster #1
                (2, 6), (6, 10), (10, 14), (14, 18), (18, 22), (22, 26), (26, 30), (30, 2), # cluster #2
                (3, 7), (7, 11), (11, 15), (15, 19), (19, 23), (23, 27), (27, 31), (31, 3), # cluster #3
            ],
            # fmt: on
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_collective_permute(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    source_target_pairs: List[Tuple[int, int]],
    dtype: torch.dtype,
    request,
    device,
):
    max_id = reduce(operator.mul, mesh_shape, 1)
    if not all(pair[0] < max_id and pair[1] < max_id for pair in source_target_pairs):
        pytest.skip("Source and target pairs are out of range")

    def collective_permute(mesh_shard_in: Operand, builder: TTIRBuilder):
        return builder.collective_permute(
            mesh_shard_in,
            source_target_pairs=source_target_pairs,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, collective_permute)

    compile_and_execute_ttir(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (32, 64),
        (32, 64, 128),
        (8, 8, 64, 64),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("split_dim", range(4))
@pytest.mark.parametrize("concat_dim", range(4))
@pytest.mark.parametrize(
    "mesh_shape, replica_groups",
    [
        ((1, 8), ((0, 1, 2, 3, 4, 5, 6, 7),)),
        ((2, 4), ((0, 4), (1, 5), (2, 6), (3, 7))),
        ((2, 4), ((0, 1, 2, 3), (4, 5, 6, 7))),
        ((4, 2), ((0, 2, 4, 6), (1, 3, 5, 7))),
        ((4, 2), ((0, 1), (2, 3), (4, 5), (6, 7))),
        ((1, 2), ((0, 1),)),
        ((2, 1), ((0, 1),)),
        ((1, 32), range(32)),
        (
            (8, 4),
            (
                (0, 1, 2, 3, 4, 5, 6, 7),
                (8, 9, 10, 11, 12, 13, 14, 15),
                (16, 17, 18, 19, 20, 21, 22, 23),
                (24, 25, 26, 27, 28, 29, 30, 31),
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_to_all(
    test_shape: Shape,
    split_dim,
    concat_dim,
    mesh_shape,
    replica_groups,
    dtype: torch.dtype,
    request,
    device,
):
    split_count = len(replica_groups[0])
    if split_dim >= len(test_shape):
        pytest.skip("Split dimension is out of range")
    if concat_dim >= len(test_shape):
        pytest.skip("Concat dimension is out of range")

    def all_to_all(mesh_shard_in: Operand, builder: TTIRBuilder):
        return builder.all_to_all(
            mesh_shard_in,
            split_dim=split_dim,
            concat_dim=concat_dim,
            split_count=split_count,
            replica_groups=replica_groups,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, all_to_all)

    compile_and_execute_ttir(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (64, 32),
        (32, 128, 64),
        (8, 8, 32, 64),
        (10, 10, 30, 60),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape, replica_groups",
    [
        ((2, 4), [(0, 1, 2, 3), (4, 5, 6, 7)]),
        ((2, 4), [(0, 4), (1, 5), (2, 6), (3, 7)]),
        ((4, 2), [(0, 1), (2, 3), (4, 5), (6, 7)]),
        ((4, 2), [(0, 2, 4, 6), (1, 3, 5, 7)]),
        ((1, 8), [(0, 1, 2, 3, 4, 5, 6, 7)]),
        ((1, 2), ((0, 1),)),
        ((2, 1), ((0, 1),)),
        ((1, 32), range(32)),
        (
            (8, 4),
            (
                (0, 1, 2, 3, 4, 5, 6, 7),
                (8, 9, 10, 11, 12, 13, 14, 15),
                (16, 17, 18, 19, 20, 21, 22, 23),
                (24, 25, 26, 27, 28, 29, 30, 31),
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_collective_broadcast(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    replica_groups,
    dtype: torch.dtype,
    request,
    device,
):
    def collective_broadcast(mesh_shard_in: Operand, builder: TTIRBuilder):
        return builder.collective_broadcast(
            mesh_shard_in,
            replica_groups=replica_groups,
        )

    test_bundle = shard_wrap_factory(test_shape, mesh_shape, collective_broadcast)

    compile_and_execute_ttir(
        test_bundle.test_fn,
        [test_bundle.input_shape],
        [dtype],
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shapes", [[(32, 32), (32, 32), (32, 32)]], ids=["32x32"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
def test_multi_return_support(
    shapes: List[Shape], dtypes: List[torch.dtype], request, device
):
    """Test that multi-return functionality works after the builder fix."""

    def multi_return_model(
        in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
    ):
        add_result = builder.add(in0, in1)
        exp_result = builder.exp(in2)
        mult_result = builder.multiply(add_result, exp_result)

        return exp_result, mult_result

    compile_and_execute_ttir(
        multi_return_model,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shapes", [[(64, 64), (64, 64)]], ids=["64x64"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 2], ids=["f32"])
def test_triple_return_support(
    shapes: List[Shape], dtypes: List[torch.dtype], request, device
):
    """Test that returning three values works."""

    def triple_return_model(in0: Operand, in1: Operand, builder: TTIRBuilder):
        add_result = builder.add(in0, in1)
        exp_result = builder.exp(in0)
        mult_result = builder.multiply(in0, in1)

        return add_result, exp_result, mult_result

    compile_and_execute_ttir(
        triple_return_model,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )
