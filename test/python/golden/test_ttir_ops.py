# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: SYSTEM_DESC_PATH=%system_desc_path% %python %s

import inspect

from ttmlir.test_utils import compile_to_flatbuffer
from ttmlir.ttir_builder import Operand, TTIRBuilder


@compile_to_flatbuffer([(128, 128)])
def test_exp(in0: Operand, builder: TTIRBuilder):
    return builder.exp(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_expm1(in0: Operand, builder: TTIRBuilder):
    return builder.expm1(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_ceil(in0: Operand, builder: TTIRBuilder):
    return builder.ceil(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_floor(in0: Operand, builder: TTIRBuilder):
    return builder.floor(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_abs(in0: Operand, builder: TTIRBuilder):
    return builder.abs(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_logical_not(in0: Operand, builder: TTIRBuilder):
    return builder.logical_not(in0)


# TODO: uncomment once we have control over generated input types (bitwise ops
# don't support floats)
# @compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
# def test_bitwise_not(in0: Operand, builder: TTIRBuilder):
#    return builder.bitwise_not(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_neg(in0: Operand, builder: TTIRBuilder):
    return builder.neg(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_sign(in0: Operand, builder: TTIRBuilder):
    return builder.sign(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_sin(in0: Operand, builder: TTIRBuilder):
    return builder.sin(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_cos(in0: Operand, builder: TTIRBuilder):
    return builder.cos(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_tan(in0: Operand, builder: TTIRBuilder):
    return builder.tan(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_tanh(in0: Operand, builder: TTIRBuilder):
    return builder.tanh(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_log(in0: Operand, builder: TTIRBuilder):
    return builder.log(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_log1p(in0: Operand, builder: TTIRBuilder):
    return builder.log1p(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_relu(in0: Operand, builder: TTIRBuilder):
    return builder.relu(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_gelu(in0: Operand, builder: TTIRBuilder):
    return builder.gelu(in0)


# TODO: figured out required param `parameter` in `LeakyReluOp.__init__`
# @compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
# def test_leaky_relu(in0: Operand, builder: TTIRBuilder):
#    return builder.leaky_relu(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_sqrt(in0: Operand, builder: TTIRBuilder):
    return builder.sqrt(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_rsqrt(in0: Operand, builder: TTIRBuilder):
    return builder.rsqrt(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_sigmoid(in0: Operand, builder: TTIRBuilder):
    return builder.sigmoid(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_reciprocal(in0: Operand, builder: TTIRBuilder):
    return builder.reciprocal(in0)


@compile_to_flatbuffer([(128, 128)], targets=["ttnn"])
def test_is_finite(in0: Operand, builder: TTIRBuilder):
    return builder.is_finite(in0)


@compile_to_flatbuffer(
    [
        (64, 128),
        (64, 128),
    ],
)
def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.add(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
)
def test_multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
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


# TODO: uncomment once we have control over generated input types (bitwise ops
# don't support floats)
# @compile_to_flatbuffer(
#    [
#        (64, 64),
#        (64, 64),
#    ],
#    targets=["ttnn"],
# )
# def test_bitwise_and(in0: Operand, in1: Operand, builder: TTIRBuilder):
#    return builder.bitwise_and(in0, in1)
#
#
# @compile_to_flatbuffer(
#    [
#        (64, 64),
#        (64, 64),
#    ],
#    targets=["ttnn"],
# )
# def test_bitwise_or(in0: Operand, in1: Operand, builder: TTIRBuilder):
#    return builder.bitwise_or(in0, in1)
#
#
# @compile_to_flatbuffer(
#    [
#        (64, 64),
#        (64, 64),
#    ],
#    targets=["ttnn"],
# )
# def test_bitwise_xor(in0: Operand, in1: Operand, builder: TTIRBuilder):
#    return builder.bitwise_xor(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_subtract(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.subtract(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_eq(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.eq(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_ne(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ne(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_ge(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ge(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_gt(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.gt(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_le(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.le(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_lt(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.lt(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_div(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.div(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_remainder(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.remainder(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_maximum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.maximum(in0, in1)


@compile_to_flatbuffer(
    [
        (64, 64),
        (64, 64),
    ],
    targets=["ttnn"],
)
def test_minimum(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.minimum(in0, in1)


# TODO: uncomment when we have control over the input types
# @compile_to_flatbuffer(
#    [
#        (64, 64),
#        (64, 64),
#        (64, 64),
#    ],
#    targets=["ttnn"],
# )
# def test_where(in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder):
#    return builder.where(in0, in1, in2)


@compile_to_flatbuffer(
    [
        (32, 32),
        (32, 32),
        (32, 32),
    ],
)
def test_arbitrary_op_chain(
    in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
):
    add = builder.add(in0, in1)
    exp = builder.exp(in2)
    return builder.multiply(add, exp)


@compile_to_flatbuffer(
    [
        (1, 784),
        (784, 256),
        (1, 256),
        (256, 10),
        (1, 10),
    ],
    targets=["ttnn"],
)
def test_mnist(
    in0: Operand,  # Input 28x28 image
    in1: Operand,  # Weight 1
    in2: Operand,  # Bias 1
    in3: Operand,  # Weight 2
    in4: Operand,  # Bias 2
    builder: TTIRBuilder,
):
    matmul_1 = builder.matmul(in0, in1)
    add_2 = builder.add(matmul_1, in2)
    relu_3 = builder.relu(add_2)
    matmul_5 = builder.matmul(relu_3, in3)
    add_6 = builder.add(matmul_5, in4)
    return builder.softmax(add_6, dimension=1)


if __name__ == "__main__":
    test_functions = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isfunction
    )

    for function_name, func in test_functions:
        if function_name.startswith("test_"):
            func()
