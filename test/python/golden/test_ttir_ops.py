# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only, get_request_kwargs

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir, build_module
from builder.base.builder_enums import *
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
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def hoisted_logical_not_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return logical_not(
                in0, builder, shape, dtype, unit_attrs=["ttir.should_hoist"]
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
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
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32, torch.float32])
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
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


def gt(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    return builder.gt(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_div(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.float32, torch.float32])
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

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_div(shape: Shape, dtype: torch.dtype, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def hoisted_div_wrapper(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            golden_input_tensor = torch.randn(shape, dtype=dtype)
            div0 = builder.div(in0, in1, unit_attrs=["ttir.should_hoist"])
            builder.set_goldens(
                {in0: golden_input_tensor, in1: golden_input_tensor},
                {div0: torch.div(golden_input_tensor, golden_input_tensor)},
            )
            return div0

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(1, 1, 32)], ids=shape_str)
@pytest.mark.parametrize("broadcast_dimensions", [[1, 16, 1]])
def test_broadcast(shape: List[int], broadcast_dimensions: List[int], request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        # Create a wrapper function that captures broadcast_dimensions
        def broadcast(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.broadcast(
                in0, broadcast_dimensions=broadcast_dimensions, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
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
    def module(builder: TTIRBuilder):
        @builder.func(shapes, input_dtypes)
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
        module,
        **get_request_kwargs(request),
        device=device,
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
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32, torch.float32, torch.float32])
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
        module,
        argument_types_string="conv2d_consteval=input,parameter,parameter",
        **get_request_kwargs(request),
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

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32, torch.float32, torch.float32])
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

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
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
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
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
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (2, 16, 32, 32),  # input (NCHW format)
            (16,),  # scale
            (16,),  # offset
            (16,),  # running_mean
            (16,),  # running_variance
        ],
        [
            (4, 32, 64, 64),  # input (NCHW format)
            (32,),  # scale
            (32,),  # offset
            (32,),  # running_mean
            (32,),  # running_variance
        ],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 5])
@pytest.mark.parametrize("dimension", [1])  # channel dimension
@pytest.mark.parametrize("epsilon", [1e-5])
@pytest.mark.parametrize("momentum", [0.1])
def test_batch_norm_training(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    dimension: int,
    epsilon: float,
    momentum: float,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def batch_norm_training(
            in0: Operand,
            scale: Operand,
            offset: Operand,
            running_mean: Operand,
            running_variance: Operand,
            builder,
            unit_attrs: Optional[List[str]] = None,
        ):

            return builder.batch_norm_training(
                in0,
                scale,
                offset,
                running_mean,
                running_variance,
                epsilon=epsilon,
                dimension=dimension,
                momentum=momentum,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes",
    [
        [
            # Matches working silicon test: embedding_backward.mlir
            (1, 32),  # input (indices) - batch=1, seq_len divisible by TILE_WIDTH (32)
            (512, 128),  # weight (vocab_size, embedding_dim)
            (1, 32, 128),  # in_gradient (indices_shape + embedding_dim)
        ],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.int32, torch.float32, torch.float32]])
def test_embedding_backward(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def embedding_backward(
            input: Operand,
            weight: Operand,
            in_gradient: Operand,
            builder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Generate valid indices within [0, vocab_size) range
            vocab_size = shapes[1][0]  # First dim of weight tensor
            valid_indices = torch.randint(0, vocab_size, shapes[0], dtype=torch.int32)
            builder.set_goldens(inputs={input: valid_indices})

            return builder.embedding_backward(
                input,
                weight,
                in_gradient,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 64)], ids=shape_str)
@pytest.mark.parametrize("dim,begin,end,step", [(0, 0, 3, 1)])
def test_index(
    shape: Shape, dim: int, begin: int, end: int, step: int, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def index(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.index(
                in0, dim=dim, begin=begin, end=end, step=step, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shape", [(4, 4)], ids=shape_str)
@pytest.mark.parametrize("dim,begin,length,stride", [(1, 2, 2, 2)])
def test_select(
    shape: Shape, dim: int, begin: int, length: int, stride: int, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
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

    compile_and_execute_ttir(module, device=device, **get_request_kwargs(request))


# TODO (ctod): These three nullary tensor creation ops can probably be combined in some way.
@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int32], ids=["bf16", "f32", "i32"]
)
def test_zeros(shape: Shape, dtype: torch.dtype, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([], [])
        def zeros(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
            return builder.zeros(shape, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=["128x128"])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32, torch.int32], ids=["bf16", "f32", "i32"]
)
def test_ones(shape: Shape, dtype: torch.dtype, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([], [])
        def ones(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
            return builder.ones(shape, dtype, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("low,high,seed", [(0.0, 1.0, 0)])
def test_rand(
    shape: Shape,
    dtype: torch.dtype,
    low: float,
    high: float,
    seed: int,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([], [])
        def rand(builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
            return builder.rand(
                shape, dtype, low=low, high=high, seed=seed, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("prob,scale,seed", [(0.2, 1.25, 2137)])
@pytest.mark.parametrize("target", ["ttnn", "emitpy", "emitc"])
def test_dropout(
    shape: Shape,
    dtype: torch.dtype,
    prob: float,
    scale: float,
    seed: int,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def dropout(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.dropout(
                in0, prob=prob, scale=scale, seed=seed, unit_attrs=unit_attrs
            )

    disable_golden = target in ["emitpy", "emitc"]
    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        device=device,
        target=target,
        disable_golden=disable_golden,
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

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def test_with_basic_callables(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_goldens({in0: torch.zeros, in1: torch.ones})
            result = builder.add(in0, in1)
            return result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_zeros(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def test_with_zeros_init(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            builder.set_goldens({in0: torch.zeros})
            zeros_result = builder.neg(in0, unit_attrs=unit_attrs)
            return zeros_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32), (64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_ones(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def test_with_ones_init(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            builder.set_goldens({in0: torch.ones})
            ones_result = builder.neg(in0, unit_attrs=unit_attrs)
            return ones_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(64, 64), (128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_eye(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
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
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_mixed(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def test_with_mixed_init(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_goldens({in0: torch.zeros, in1: torch.ones})
            add_result = builder.add(in0, in1)
            return add_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(16, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_callable_initialization_custom_lambda(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def test_with_custom_lambda(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            custom_init = lambda s: torch.full(s, 2.0)
            builder.set_goldens({in0: custom_init})
            result = builder.multiply(in0, in0)
            return result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
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


@pytest.mark.parametrize("shape", [(256, 128, 2, 2)], ids=shape_str)
@pytest.mark.parametrize("dims", [[2, 3]])
def test_reverse(shape: Shape, dims: List[int], request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def reverse(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.reverse(in0, dimensions=dims, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shapes", [[(10, 64, 32, 3), (10, 128, 128, 3)]], ids=shapes_list_str
)
@pytest.mark.parametrize("scale_factor", [[2, 4]])
def test_upsample2d(shapes: List[Shape], scale_factor: List[int], request, device):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
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
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "shape,dtype,start,end,step,dim",
    [
        ((5,), torch.float32, 0, 5, 1, 0),
        ((5, 3), torch.int64, 0, 5, 1, 0),
        ((5, 3), torch.int64, 0, 3, 1, 1),
    ],
)
def test_arange(
    shape: Shape,
    dtype: torch.dtype,
    start: int,
    end: int,
    step: int,
    dim: int,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def arange(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.arange(
                shape, dtype, start, end, step, dim, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shapes", [[(4, 4, 128, 128)]], ids=shapes_list_str)
@pytest.mark.parametrize("dim", [1])
def test_cumsum(shapes: List[Shape], dim: int, request, device):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def cumsum(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.cumsum(in0, dim=dim, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


def prod(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.prod(in0, [1], False, unit_attrs=unit_attrs)


@pytest.mark.xfail(reason="Fails Golden")
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 64, 512), (1, 32, 3, 512)]], ids=shapes_list_str
)
def test_fill_cache(shapes: List[Shape], request, device):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def fill_cache(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.fill_cache(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.xfail(reason="run error")
@pytest.mark.parametrize(
    "shapes", [[(1, 32, 64, 512), (1, 32, 1, 512), (1,)]], ids=shapes_list_str
)
@pytest.mark.parametrize("dtypes", [[torch.float32, torch.float32, torch.int32]])
def test_update_cache(shapes: List[Shape], dtypes: List[torch.dtype], request, device):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def update_cache(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.update_cache(in0, in1, in2, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


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
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def quantize(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.quantize(
                in0, scale, zero_point, dtype, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
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
    def module(builder: TTIRBuilder):
        @builder.func([shape], [input_dtype])
        def dequantize(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.dequantize(
                in0, scale, zero_point, dtype, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
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
    def module(builder: TTIRBuilder):
        @builder.func([shape], [input_dtype])
        def requantize(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.requantize(
                in0, scale, zero_point, dtype, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
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


# Test hoisted permute separately because it requires unique input shapes.
@x86_only
@pytest.mark.parametrize(
    "shapes,permutation",
    [
        ([(2, 3, 4)], [2, 0, 1]),
        ([(128, 128)], [0, 1]),
        ([(128, 64, 32)], [2, 0, 1]),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_permute(shapes, permutation, request, target: str, device):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32])
        def permute(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.permute(in0, permutation, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "shape,begins,ends,step,dtype",
    [
        ((64, 64), [0, 0], [32, 32], None, torch.float32),
        ((128, 128), [10, 20], [50, 60], [1, 1], torch.int32),
        ((32, 64, 64), [5, 10, 15], [25, 50, 55], [2, 2, 1], torch.int32),
    ],
    ids=["basic_slice_f32", "explicit_step_f32", "3d_slice_i32"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_slice(
    shape: Shape,
    dtype: torch.dtype,
    begins: List[int],
    ends: List[int],
    step: List[int],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def slice(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            # Now use the slice operation with the CPU hoisting attribute
            return builder.slice(
                in0, begins, ends, step, unit_attrs=["ttir.should_hoist"]
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Add test for hoisted where operation
@x86_only
@pytest.mark.parametrize(
    "shapes", [[(64, 64), (64, 64), (64, 64)]], ids=shapes_list_str
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_where(shapes, request, target: str, device):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def where(condition: Operand, x: Operand, y: Operand, builder: TTIRBuilder):
            return builder.where(condition, x, y, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "input_shape,output_shape,dtype",
    [
        # (input_shape, output_shape, dtype)
        ((2, 3, 4), (24,), torch.float32),
        ((128, 128), (16384,), torch.float32),
        ((128, 64, 32), (128, 2048), torch.int32),
    ],
    ids=["3d_to_1d_f32", "2d_to_1d_f32", "3d_to_2d_i32"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_reshape(
    input_shape, output_shape, dtype, request, target: str, device
):
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def reshape(in0: Operand, builder: TTIRBuilder):
            return builder.reshape(in0, output_shape, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "input_shape,dims,dtype",
    [
        # (input_shape, permutation)
        ((2, 3, 4), [2, 1], torch.float32),
        ((128, 128), [1, 0], torch.float32),
        ((128, 64, 32), [0, 2], torch.int32),
    ],
    ids=["3d_perm_f32", "2d_perm_f32", "3d_perm_i32"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_transpose(input_shape, dims, dtype, request, target: str, device):
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def hoisted_transpose(in0: Operand, builder: TTIRBuilder):
            # For 2D tensors with permutation [1, 0], swap dimensions 0 and 1
            # For 3D tensors with permutation [2, 1, 0], swap dimensions 0 and 2
            dim0 = dims[0]
            dim1 = dims[1]
            return builder.transpose(
                in0, dim0=dim0, dim1=dim1, unit_attrs=["ttir.should_hoist"]
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize("shape", [(1, 128, 128)], ids=shape_str)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_squeeze(shape: Shape, dim: int, target: str, request, device):
    """Test hoisted squeeze operation with appropriate shape that has a dimension of size 1"""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_squeeze(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.squeeze(in0, dim=dim, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
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
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=pipeline_options,
    )


@pytest.mark.parametrize(
    "inputs_shapes,inputs_dtypes",
    [
        pytest.param(
            [(33, 32), (512, 128)],
            [torch.float32] * 2,
            marks=[pytest.mark.skip_config(["ttmetal"])],
        ),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_unique_ops(
    inputs_shapes: List[Shape],
    inputs_dtypes: List[torch.dtype],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(inputs_shapes, inputs_dtypes)
        def embedding(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.embedding(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "indices_shape,weight_shape",
    [
        (
            (32, 32),
            (512, 128),
        ),  # 2D indices: (batch, seq_len), weight: (vocab, embed_dim)
        ((64,), (256, 64)),  # 1D indices: (seq_len,), smaller vocab and embed_dim
        ((1, 64), (1024, 256)),  # Single batch, larger vocab and embed_dim
        ((8, 128), (512, 64)),  # Different batch and seq_len
        (
            (2, 4),
            (1, 1, 10, 10),
        ),  # 2D indices, 4D weight (effectively 2D with leading singletons)
    ],
    ids=["2d_basic", "1d_indices", "large_vocab", "varied_dims", "4d_weight"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_embedding(
    indices_shape: Shape,
    weight_shape: Shape,
    target: str,
    request,
    device,
):
    """Test the hoisted embedding operation."""
    # Vocab size is at second-to-last dimension for "effectively 2D" weights.
    vocab_size = weight_shape[-2]

    def module(builder: TTIRBuilder):
        @builder.func([indices_shape, weight_shape], [torch.float32, torch.float32])
        def hoisted_embedding(
            indices: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Generate valid indices within [0, vocab_size) range.
            valid_indices = torch.randint(
                0, vocab_size, indices_shape, dtype=torch.float32
            )
            builder.set_goldens(inputs={indices: valid_indices})
            return builder.embedding(indices, weight, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize("shape", [(4, 4)], ids=shape_str)
@pytest.mark.parametrize("dim_args", [[0]])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_reduce_or(
    shape: Shape, dim_args: List[int], target: str, request, device
):
    """Test the hoisted reduce_or operation with proper dimensions and keep_dim parameter"""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_reduce_or_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.reduce_or(
                in0, dim_arg=dim_args, keep_dim=True, unit_attrs=["ttir.should_hoist"]
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
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

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_broadcast(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.broadcast(
                in0, broadcast_dims, unit_attrs=["ttir.should_hoist"]
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
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
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [input_dtype])
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
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
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
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
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
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
    [
        # Standard matrix multiplication: [M, K] x [K, N] -> [M, N]
        ([(10, 20), (20, 30)], [], [1], [], [0]),
        # Batched matrix multiplication: [B, M, K] x [B, K, N] -> [B, M, N]
        ([(5, 10, 20), (5, 20, 30)], [0], [2], [0], [1]),
        # 3D tensor @ 2D tensor: [B, M, K] x [K, N] -> [B, M, N]
        ([(5, 10, 20), (20, 30)], [], [2], [], [0]),
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
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def dot_general_wrapper(
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
                unit_attrs=["ttir.should_hoist"],
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
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

    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
        def mesh_shard_devices(in0: Operand, builder: TTIRBuilder):
            mesh_shard_in0 = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
            neg_output = builder.neg(mesh_shard_in0)
            return builder.mesh_shard(
                neg_output,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
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

    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_gather(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            all_gather0 = builder.all_gather(
                in_shard,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
            )

            return builder.mesh_shard(
                all_gather0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize(
    "test_shape",
    [
        (1, 1, 256, 256),
        (256, 256, 1, 1),
        (1, 64, 64),
        (64, 64),
        (64, 65),
        (65, 64),
        (32, 64),
        (33, 65),
        (1, 1, 1, 1, 1, 1, 32, 256, 256),
        (1, 1, 32, 256, 256),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize(
    "mesh_shape, cluster_axis",
    [
        ((1, 2), 1),
        ((1, 8), 1),
        ((2, 4), 0),
        ((2, 4), 1),
        ((1, 32), 1),
        ((8, 4), 0),
        ((8, 4), 1),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_all_reduce(
    test_shape: Shape,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    dtype: torch.dtype,
    request,
    device,
):
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    # test 'sum' only for now. Other reduce types are not supported yet.
    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_reduce(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            all_reduce0 = builder.all_reduce(
                in_shard,
                reduce_type=ReduceType.Sum.value,
                cluster_axis=cluster_axis,
            )

            return builder.mesh_shard(
                all_reduce0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
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

    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    # test 'sum' only for now. Other reduce types are not supported yet.
    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def reduce_scatter(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            reduce_scatter0 = builder.reduce_scatter(
                in_shard,
                reduce_type=ReduceType.Sum.value,
                scatter_dim=scatter_dim,
                cluster_axis=cluster_axis,
            )

            return builder.mesh_shard(
                reduce_scatter0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
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
        ((1, 2), [(0, 1)]),
        ((1, 2), [(0, 1), (1, 0)]),
        ((2, 4), [(0, 1), (1, 2), (2, 3), (3, 0)]),
        ((2, 4), [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]),
        ((2, 4), [(0, 4), (4, 0), (1, 5), (5, 1), (2, 6), (6, 2), (3, 7), (7, 3)]),
        ((2, 4), [(0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)]),
        ((2, 4), [(0, 2), (1, 3), (4, 6), (5, 7), (2, 0), (3, 1), (6, 4), (7, 5)]),
        ((2, 4), [(0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0)]),
        ((2, 4), [(0, 1), (2, 3), (4, 5), (6, 7)]),
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
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def collective_permute_wrapper(in0: Operand, builder: TTIRBuilder):
            mesh_shard_in = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            collective_permute_out = builder.collective_permute(
                mesh_shard_in, source_target_pairs
            )

            return builder.mesh_shard(
                collective_permute_out,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
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

    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_to_all_wrapper(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            all_to_all0 = builder.all_to_all(
                in_shard,
                split_dim=split_dim,
                concat_dim=concat_dim,
                split_count=split_count,
                replica_groups=replica_groups,
            )

            return builder.mesh_shard(
                all_to_all0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
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
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def collective_broadcast(in0: Operand, builder: TTIRBuilder):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            collective_broadcast_out = builder.collective_broadcast(
                in_shard, replica_groups
            )

            return builder.mesh_shard(
                collective_broadcast_out,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        device=device,
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("shapes", [[(32, 32), (32, 32), (32, 32)]], ids=["32x32"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
def test_multi_return_support(
    shapes: List[Shape], dtypes: List[torch.dtype], request, device
):
    """Test that multi-return functionality works after the builder fix."""

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def multi_return_model(
            in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
        ):
            add_result = builder.add(in0, in1)
            exp_result = builder.exp(in2)
            mult_result = builder.multiply(add_result, exp_result)

            return exp_result, mult_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("shapes", [[(64, 64), (64, 64)]], ids=["64x64"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 2], ids=["f32"])
def test_triple_return_support(
    shapes: List[Shape], dtypes: List[torch.dtype], request, device
):
    """Test that returning three values works."""

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def triple_return_model(in0: Operand, in1: Operand, builder: TTIRBuilder):
            add_result = builder.add(in0, in1)
            exp_result = builder.exp(in0)
            mult_result = builder.multiply(in0, in1)

            return add_result, exp_result, mult_result

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_multiple_function(target, request, device):
    def my_module(builder: TTIRBuilder):
        @builder.func([(32, 32)], [torch.float32])
        def my_modela(in0: Operand, builder: TTIRBuilder):
            sigmoid0 = builder.sigmoid(in0)
            return sigmoid0

        @builder.func([(32, 32)], [torch.float32])
        def my_modelb(in0: Operand, builder: TTIRBuilder):
            sigmoid0 = builder.sigmoid(in0)
            return sigmoid0

    compile_and_execute_ttir(
        my_module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_device_cpu_module(target, request, device):
    def my_module(builder: TTIRBuilder):
        @builder.device_module
        def my_device_module(builder: TTIRBuilder):
            @builder.func([(32, 32)], [torch.float32])
            def my_modela(in0: Operand, builder: TTIRBuilder):
                sigmoid0 = builder.sigmoid(in0)
                return sigmoid0

        @builder.cpu_module
        def my_cpu_module(builder: TTIRBuilder):
            @builder.func([(32, 32)], [torch.float32])
            def my_modelb(in0: Operand, builder: TTIRBuilder):
                sigmoid0 = builder.sigmoid(in0)
                return sigmoid0

    compile_and_execute_ttir(
        my_module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_nested_function_calls(target, request, device):
    def my_module(builder: TTIRBuilder):
        @builder.device_module
        def my_device_module(builder: TTIRBuilder):
            @builder.func([(32, 32)], [torch.float32])
            def my_modela(in0: Operand, builder: TTIRBuilder):
                def nested_func(in0: Operand, builder: TTIRBuilder):
                    relu0 = builder.relu(in0)
                    return relu0

                sigmoid0 = builder.sigmoid(in0)
                nested_func0 = builder.call(nested_func, [sigmoid0])
                return nested_func0

            @builder.func([(32, 32)], [torch.float32])
            def my_modelb(in0: Operand, builder: TTIRBuilder):
                sigmoid0 = builder.sigmoid(in0)
                return sigmoid0

    compile_and_execute_ttir(
        my_module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize(
    "shape,dtype,start,end,step,dim",
    [
        ((5,), torch.float32, 0, 5, 1, 0),
        ((10,), torch.int32, 0, 10, 1, 0),
        ((8,), torch.float32, 2, 10, 1, 0),
    ],
    ids=["f32_simple", "i32_simple", "f32_offset_start"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_arange(
    shape: Shape,
    dtype: torch.dtype,
    start: int,
    end: int,
    step: int,
    dim: int,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_arange(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.arange(
                shape, dtype, start, end, step, dim, unit_attrs=["ttir.should_hoist"]
            )

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes,dim",
    [
        ([(4, 4, 32, 32)], 1),
        ([(2, 8, 16, 16)], 0),
        ([(4, 4, 32, 32)], -1),
    ],
    ids=["dim1", "dim0", "dim_negative"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_cumsum(
    shapes: List[Shape],
    dim: int,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def hoisted_cumsum(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.cumsum(in0, dim=dim, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shape,repeat_dims",
    [
        ((32, 32), [2, 1]),
        ((16, 16), [1, 3]),
        ((8, 8, 8), [2, 2, 1]),
    ],
    ids=["repeat_dim0", "repeat_dim1", "3d_repeat"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_repeat(
    shape: Shape,
    repeat_dims: List[int],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_repeat(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.repeat(in0, repeat_dims, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((32, 32), 0),
        ((32, 32), 1),
        ((32, 32), 2),
        ((16, 16, 16), 0),
    ],
    ids=["unsqueeze_dim0", "unsqueeze_dim1", "unsqueeze_dim2", "3d_unsqueeze"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_unsqueeze(
    shape: Shape,
    dim: int,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_unsqueeze(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.unsqueeze(in0, dim=dim, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shape,input_dtype,output_dtype",
    [
        ((32, 32), torch.int32, torch.float32),
        ((64, 64), torch.float32, torch.bfloat16),
    ],
    ids=["i32_to_f32", "f32_to_bf16"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_typecast(
    shape: Shape,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [input_dtype])
        def hoisted_typecast(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.typecast(in0, output_dtype, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@x86_only
@pytest.mark.parametrize(
    "shapes,dim",
    [
        ([(32, 32), (32, 32)], 0),
        ([(32, 32), (32, 32)], 1),
        ([(16, 32), (16, 32), (16, 32)], 0),
        ([(32, 16), (32, 16)], -1),
    ],
    ids=["concat_dim0", "concat_dim1", "concat_3_tensors", "concat_negative_dim"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_hoisted_concat(
    shapes: List[Shape],
    dim: int,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def hoisted_concat(
            *inputs,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder = inputs[-1]
            tensors = list(inputs[:-1])
            return builder.concat(tensors, dim=dim, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "input_shape,num_heads,transpose_key",
    [
        # MHA case: input [batch, seq, 3 * hidden_size]
        # batch=2, seq=128, hidden=256, num_heads=8, head_size=32
        ((2, 128, 768), 8, False),
        # MHA case with transpose_key
        ((2, 128, 768), 8, True),
    ],
    ids=["mha_no_transpose", "mha_transpose_key"],
)
def test_split_query_key_value_and_split_heads_mha(
    input_shape: Shape,
    num_heads: int,
    transpose_key: bool,
    request,
    device,
):
    """Test split_query_key_value_and_split_heads operation (MHA case)."""

    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
        def split_qkv(
            input_tensor: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query, key, value = builder.split_query_key_value_and_split_heads(
                input_tensor,
                num_heads=num_heads,
                transpose_key=transpose_key,
            )
            return query, key, value

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@pytest.mark.parametrize(
    "q_shape,kv_shape,num_heads,num_kv_heads,transpose_key",
    [
        # GQA case: separate Q and KV tensors
        # batch=2, seq=128, num_heads=8, num_kv_heads=2, head_size=32
        # Q: [batch, seq, num_heads * head_size] = [2, 128, 256]
        # KV: [batch, seq, 2 * num_kv_heads * head_size] = [2, 128, 128]
        ((2, 128, 256), (2, 128, 128), 8, 2, False),
        # GQA case with transpose_key
        ((2, 128, 256), (2, 128, 128), 8, 2, True),
    ],
    ids=["gqa_no_transpose", "gqa_transpose_key"],
)
def test_split_query_key_value_and_split_heads_gqa(
    q_shape: Shape,
    kv_shape: Shape,
    num_heads: int,
    num_kv_heads: int,
    transpose_key: bool,
    request,
    device,
):
    """Test split_query_key_value_and_split_heads operation (GQA case)."""

    def module(builder: TTIRBuilder):
        @builder.func([q_shape, kv_shape], [torch.float32, torch.float32])
        def split_qkv_gqa(
            q_tensor: Operand,
            kv_tensor: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query, key, value = builder.split_query_key_value_and_split_heads(
                q_tensor,
                num_heads=num_heads,
                transpose_key=transpose_key,
                kv_input_tensor=kv_tensor,
                num_kv_heads=num_kv_heads,
            )
            return query, key, value

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "input_shape,num_heads,transpose_key",
    [
        # MHA case: input [batch, seq, 3 * hidden_size]
        # batch=2, seq=128, hidden=256, num_heads=8, head_size=32
        ((2, 128, 768), 8, False),
        # MHA case with transpose_key
        ((2, 128, 768), 8, True),
    ],
    ids=["mha_no_transpose", "mha_transpose_key"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_split_query_key_value_and_split_heads_mha(
    input_shape: Shape,
    num_heads: int,
    transpose_key: bool,
    target: str,
    request,
    device,
):
    """Test split_query_key_value_and_split_heads operation (MHA case) with CPU hoisting."""

    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
        def split_qkv(
            input_tensor: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query, key, value = builder.split_query_key_value_and_split_heads(
                input_tensor,
                num_heads=num_heads,
                transpose_key=transpose_key,
                unit_attrs=["ttir.should_hoist"],
            )
            return query, key, value

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@x86_only
@pytest.mark.parametrize(
    "q_shape,kv_shape,num_heads,num_kv_heads,transpose_key",
    [
        # GQA case: separate Q and KV tensors
        # batch=2, seq=128, num_heads=8, num_kv_heads=2, head_size=32
        ((2, 128, 256), (2, 128, 128), 8, 2, False),
        # GQA case with transpose_key
        ((2, 128, 256), (2, 128, 128), 8, 2, True),
    ],
    ids=["gqa_no_transpose", "gqa_transpose_key"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_hoisted_split_query_key_value_and_split_heads_gqa(
    q_shape: Shape,
    kv_shape: Shape,
    num_heads: int,
    num_kv_heads: int,
    transpose_key: bool,
    target: str,
    request,
    device,
):
    """Test split_query_key_value_and_split_heads operation (GQA case) with CPU hoisting."""

    def module(builder: TTIRBuilder):
        @builder.func([q_shape, kv_shape], [torch.float32, torch.float32])
        def split_qkv_gqa(
            q_tensor: Operand,
            kv_tensor: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query, key, value = builder.split_query_key_value_and_split_heads(
                q_tensor,
                num_heads=num_heads,
                transpose_key=transpose_key,
                kv_input_tensor=kv_tensor,
                num_kv_heads=num_kv_heads,
                unit_attrs=["ttir.should_hoist"],
            )
            return query, key, value

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
