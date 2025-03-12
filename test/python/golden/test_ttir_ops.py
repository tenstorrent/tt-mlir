# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import pytest
import inspect
import torch

from typing import Callable, List, Optional

from ttmlir.test_utils import compile_to_flatbuffer, set_output_path, compile_as_mlir_module, ttir_to_ttmetal, ttir_to_ttnn, ttmetal_to_flatbuffer, ttnn_to_flatbuffer
from ttmlir.ttir_builder import Operand, TTIRBuilder, Attribute, UnitAttr, Shape
from ttmlir.dialects import ttir
from ttmlir.passes import  MLIRModuleLogger

# NOTE: This test is not valid for TTRT Perf due to weird issues with perf collection
"""
@compile_to_flatbuffer([(1, 128, 128, 1)], targets=["ttnn"])
def test_squeeze(in0: Operand, builder: TTIRBuilder):
    return builder.squeeze(in0, 0)
"""

# NOTE: Same as Squeeze, this Op is not valid for TTRT Perf.
"""
@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_unsqueeze(in0: Operand, builder: TTIRBuilder):
    return builder.unsqueeze(in0, 0)
"""

def exp(in0: Operand, builder: TTIRBuilder):
    return builder.exp(in0)


def expm1(in0: Operand, builder: TTIRBuilder):
    return builder.expm1(in0)


def ceil(in0: Operand, builder: TTIRBuilder):
    return builder.ceil(in0)


def floor(in0: Operand, builder: TTIRBuilder):
    return builder.floor(in0)


def abs(in0: Operand, builder: TTIRBuilder):
    return builder.abs(in0)


def logical_not(in0: Operand, builder: TTIRBuilder):
    return builder.logical_not(in0)


# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking
"""
@compile_to_flatbuffer([(128, 128)], inputs_types=[torch.int8], targets=["ttnn"])
def test_bitwise_not(in0: Operand, builder: TTIRBuilder):
    return builder.bitwise_not(in0)
"""


def neg(in0: Operand, builder: TTIRBuilder):
    return builder.neg(in0)


def sign(in0: Operand, builder: TTIRBuilder):
    return builder.sign(in0)


def sin(in0: Operand, builder: TTIRBuilder):
    return builder.sin(in0)


def cos(in0: Operand, builder: TTIRBuilder):
    return builder.cos(in0)


def tan(in0: Operand, builder: TTIRBuilder):
    return builder.tan(in0)


def tanh(in0: Operand, builder: TTIRBuilder):
    return builder.tanh(in0)


def log(in0: Operand, builder: TTIRBuilder):
    return builder.log(in0)


def log1p(in0: Operand, builder: TTIRBuilder):
    return builder.log1p(in0)


def relu(in0: Operand, builder: TTIRBuilder):
    return builder.relu(in0)


def gelu(in0: Operand, builder: TTIRBuilder):
    return builder.gelu(in0)


def clamp(in0: Operand, builder: TTIRBuilder):
    return builder.clamp(in0, max_arg=1.0, min_arg=0.0)


def leaky_relu(in0: Operand, builder: TTIRBuilder):
    return builder.leaky_relu(in0)


def sqrt(in0: Operand, builder: TTIRBuilder):
    return builder.sqrt(in0)


def cbrt(in0: Operand, builder: TTIRBuilder):
    return builder.cbrt(in0)


def rsqrt(in0: Operand, builder: TTIRBuilder):
    return builder.rsqrt(in0)


def sigmoid(in0: Operand, builder: TTIRBuilder):
    return builder.sigmoid(in0)


def reciprocal(in0: Operand, builder: TTIRBuilder):
    return builder.reciprocal(in0)


def is_finite(in0: Operand, builder: TTIRBuilder):
    return builder.is_finite(in0)


@compile_to_flatbuffer(
    [
        (64, 128),
        (32, 128),
        (16, 128),
    ],
    targets=["ttnn"],
)
def test_concat(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
    return builder.concat([in0, in1, in2])


def add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.add(in0, in1)


def multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.multiply(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_logical_and(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_and(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_logical_or(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_or(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_logical_xor(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_xor(in0, in1)


# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking

"""
@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    inputs_types=[torch.int8, torch.int8],
    targets=["ttnn"],
)
def test_bitwise_and(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_and(in0, in1)
"""

# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking

"""
@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    inputs_types=[torch.int8, torch.int8],
    targets=["ttnn"],
)
def test_bitwise_or(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_or(in0, in1)
"""

# NOTE: The generated flatbuffer will currently fail to run due to only floats
# being supported by the runtime. See issue #1775 for tracking

"""
@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    inputs_types=[torch.int8, torch.int8],
    targets=["ttnn"],
)
def test_bitwise_xor(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.bitwise_xor(in0, in1)
"""


def subtract(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.subtract(in0, in1)


def eq(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.eq(in0, in1)


def ne(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ne(in0, in1)


def ge(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ge(in0, in1)


def gt(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.gt(in0, in1)


def le(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.le(in0, in1)


def lt(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.lt(in0, in1)


def div(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.div(in0, in1)


def remainder(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.remainder(in0, in1)


def maximum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.maximum(in0, in1)


def minimum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.minimum(in0, in1)


def power(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.power(in0, in1)


def matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.matmul(in0, in1)


def sum(in0: Operand, builder: TTIRBuilder):
    return builder.sum(in0)


def mean(in0: Operand, builder: TTIRBuilder):
    return builder.mean(in0)

def max(in0: Operand, builder: TTIRBuilder):
    return builder.max(in0)


def min(in0: Operand, builder: TTIRBuilder):
    return builder.min(in0)


@compile_to_flatbuffer(
    [
        (32, 64),
    ],
    targets=["ttnn"],
)
def test_reshape(in0: Operand, builder: TTIRBuilder):
    return builder.reshape(in0, [2048])


@compile_to_flatbuffer(
    [
        (32, 64),
    ],
    targets=["ttnn"],
)
def test_transpose(in0: Operand, builder: TTIRBuilder):
    return builder.transpose(in0)


# @compile_to_flatbuffer(
#   [
#       (64, 64),
#       (64, 64),
#       (64, 64),
#   ],
#   inputs_types = [torch.int8, torch.float32, torch.float32],
#   targets=["ttnn"],
# )
# def test_where(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
#   return builder.where(in0, in1, in2)
#


@compile_to_flatbuffer(
    [
        (32, 32),
        (32, 32),
        (32, 32),
    ],
    targets=["ttnn"],
)
def test_arbitrary_op_chain(
    in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
):
    add = builder.add(in0, in1)
    exp = builder.exp(in2)
    return builder.multiply(add, exp)


def hoisted_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    # Use op_proxy directly since it accepts ttir_kwargs
    return builder.op_proxy(
        torch.add,
        ttir.AddOp,
        [in0, in1],
        unit_attrs={"should_hoist": UnitAttr.get(builder._ctx)},
        use_zeros=True,
    )

unary_ops = [exp, expm1, floor, abs, logical_not, neg, sign, cos, sin, tan,
             tanh, log,log1p, relu, gelu, clamp, leaky_relu, sqrt, cbrt, rsqrt,
             sigmoid, reciprocal, is_finite, ceil, sum, mean, max, min]

@pytest.mark.parametrize("shape", [(128,128)], ids=str)
@pytest.mark.parametrize("test_fn", unary_ops)
def test_unary_ops_to_flatbuffer(
    test_fn: Callable,
    shape: Shape,
    request):
    _compile_to_flatbuffer(test_fn, inputs_shapes=[shape], test_name=request.node.name)


@pytest.mark.parametrize("shape", [(128,128)], ids=str)
@pytest.mark.parametrize("test_fn", [add, multiply, subtract, eq, ne, le, lt,
                                     ge, gt, div, remainder, maximum, minimum,
                                     power, matmul, hoisted_add])
def test_binary_ops_to_flatbuffer(
    test_fn: Callable,
    shape: Shape,
    request):
    """ NOTE: this function is _only_ for binary ops that take the same shape arguments
    """
    _compile_to_flatbuffer(test_fn, inputs_shapes=[shape, shape], test_name=request.node.name)


def _compile_to_flatbuffer(
    test_fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[torch.dtype]] = None,
    test_name: Optional[str] = None,
    module_dump: bool = False,
):
    # Snoop the name of `test_fn` if no override to the test name is provided
    if test_name is None:
        test_base = test_fn.__name__
    else:
        test_base = test_name

    module, builder = compile_as_mlir_module(
        test_fn, inputs_shapes, inputs_types
    )

    if module_dump:
        with open(test_base + "_ttir.mlir", "w") as f:
            f.write(str(module))

    module_logger = MLIRModuleLogger()
    module_logger.attach_context(module.context)
    module = ttir_to_ttnn(module, module_dump, test_base + "_ttnn.mlir")
    ttnn_to_flatbuffer(
        module, builder, test_base + ".ttnn", module_logger.module_log
    )

    # TODO: execute the flatbuffer here to check goldens
