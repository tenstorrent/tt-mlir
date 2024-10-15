# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# XFAIL: true
# RUN: %python %s | FileCheck %s

from ttmlir.ttir_builder import TTIRBuilder, compile_as_mlir_module, Operand


@compile_as_mlir_module((32, 32), (32, 32))
def test_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
    # CHECK: %0 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %1 = "ttir.add"(%arg0, %arg1, %0)
    # CHECK: return %1 : tensor<32x32xf32>

    return builder.add(in0, in1)


@compile_as_mlir_module((64, 64), (64, 64))
def test_multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
    # CHECK: %0 = tensor.empty() : tensor<64x64xf32>
    # CHECK: %1 = "ttir.multiply"(%arg0, %arg1, %0)
    # CHECK: return %1 : tensor<64x64xf32>

    return builder.multiply(in0, in1)


@compile_as_mlir_module((128, 128))
def test_exp(in0: Operand, builder: TTIRBuilder):
    # CHECK: %0 = tensor.empty() : tensor<128x128xf32>
    # CHECK: %1 = "ttir.exp"(%arg0, %0)
    # CHECK: return %1 : tensor<128x128xf32>

    return builder.exp(in0)


@compile_as_mlir_module((32, 32), (32, 32), (32, 32))
def test_arbitrary_op_chain(
    in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
):
    # CHECK: %0 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %1 = "ttir.add"(%arg0, %arg1, %0)
    # CHECK: %2 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %3 = "ttir.exp"(%arg2, %2)
    # CHECK: %4 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %5 = "ttir.multiply"(%1, %3, %4)
    # CHECK: %6 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %7 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %8 = "ttir.multiply"(%5, %6, %7)
    # CHECK: return %8 : tensor<32x32xf32>

    add = builder.add(in0, in1)
    exp = builder.exp(in2)
    mul = builder.multiply(add, exp)
    in3 = builder.empty(builder.get_shape(mul))
    return builder.multiply(mul, in3)


if __name__ == "__main__":
    test_add()
    test_multiply()
    test_exp()
    test_arbitrary_op_chain()
