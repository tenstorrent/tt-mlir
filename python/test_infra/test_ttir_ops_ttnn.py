# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s

import inspect
import os

from ttmlir.test_utils import (
    compile_as_mlir_module,
    ttnn_to_flatbuffer,
    ttir_to_ttnn,
)
from ttmlir.ttir_builder import Operand, TTIRBuilder

system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")


@ttnn_to_flatbuffer(output_file_name="test_exp.ttnn")
@ttir_to_ttnn(output_file_name="test_exp.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((128, 128))
def test_exp_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.exp(in0)


@ttnn_to_flatbuffer(output_file_name="test_abs.ttnn")
@ttir_to_ttnn(output_file_name="test_abs.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((128, 128))
def test_abs_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.abs(in0)


@ttnn_to_flatbuffer(output_file_name="test_logical_not.ttnn")
@ttir_to_ttnn(
    output_file_name="test_logical_not.mlir",
    system_desc_path=f"{system_desc_path}",
)
@compile_as_mlir_module((128, 128))
def test_logical_not_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.logical_not(in0)


@ttnn_to_flatbuffer(output_file_name="test_neg.ttnn")
@ttir_to_ttnn(output_file_name="test_neg.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((128, 128))
def test_neg_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.neg(in0)


@ttnn_to_flatbuffer(output_file_name="test_relu.ttnn")
@ttir_to_ttnn(output_file_name="test_relu.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((128, 128))
def test_relu_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.relu(in0)


@ttnn_to_flatbuffer(output_file_name="test_sqrt.ttnn")
@ttir_to_ttnn(output_file_name="test_sqrt.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((128, 128))
def test_sqrt_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.sqrt(in0)


@ttnn_to_flatbuffer(output_file_name="test_rsqrt.ttnn")
@ttir_to_ttnn(
    output_file_name="test_rsqrt.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((128, 128))
def test_rsqrt_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.rsqrt(in0)


@ttnn_to_flatbuffer(output_file_name="test_sigmoid.ttnn")
@ttir_to_ttnn(
    output_file_name="test_sigmoid.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((128, 128))
def test_sigmoid_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.sigmoid(in0)


@ttnn_to_flatbuffer(output_file_name="test_reciprocal.ttnn")
@ttir_to_ttnn(
    output_file_name="test_reciprocal.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((128, 128))
def test_reciprocal_ttnn(in0: Operand, builder: TTIRBuilder):
    return builder.reciprocal(in0)


@ttnn_to_flatbuffer(output_file_name="test_add.ttnn")
@ttir_to_ttnn(output_file_name="test_add.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((64, 128), (64, 128))
def test_add_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.add(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_multiply.ttnn")
@ttir_to_ttnn(
    output_file_name="test_multiply.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((64, 64), (64, 64))
def test_multiply_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.multiply(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_logical_and.ttnn")
@ttir_to_ttnn(
    output_file_name="test_logical_and.mlir",
    system_desc_path=f"{system_desc_path}",
)
@compile_as_mlir_module((64, 64), (64, 64))
def test_logical_and_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_and(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_logical_or.ttnn")
@ttir_to_ttnn(
    output_file_name="test_logical_or.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((64, 64), (64, 64))
def test_logical_or_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.logical_or(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_subtract.ttnn")
@ttir_to_ttnn(
    output_file_name="test_subtract.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((64, 64), (64, 64))
def test_subtract_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.subtract(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_eq.ttnn")
@ttir_to_ttnn(output_file_name="test_eq.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((64, 64), (64, 64))
def test_eq_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.eq(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_ne.ttnn")
@ttir_to_ttnn(output_file_name="test_ne.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((64, 64), (64, 64))
def test_ne_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ne(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_ge.ttnn")
@ttir_to_ttnn(output_file_name="test_ge.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((64, 64), (64, 64))
def test_ge_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.ge(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_gt.ttnn")
@ttir_to_ttnn(output_file_name="test_gt.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((64, 64), (64, 64))
def test_gt_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.gt(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_le.ttnn")
@ttir_to_ttnn(output_file_name="test_le.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((64, 64), (64, 64))
def test_le_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.le(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_lt.ttnn")
@ttir_to_ttnn(output_file_name="test_lt.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((64, 64), (64, 64))
def test_lt_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.lt(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_div.ttnn")
@ttir_to_ttnn(output_file_name="test_div.mlir", system_desc_path=f"{system_desc_path}")
@compile_as_mlir_module((64, 64), (64, 64))
def test_div_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.div(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_maximum.ttnn")
@ttir_to_ttnn(
    output_file_name="test_maximum.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((64, 64), (64, 64))
def test_maximum_ttnn(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.maximum(in0, in1)


@ttnn_to_flatbuffer(output_file_name="test_arbitrary_op_chain.ttnn")
@ttir_to_ttnn(
    output_file_name="test_arbitrary_op_chain.mlir",
    system_desc_path=f"{system_desc_path}",
)
@compile_as_mlir_module((32, 32), (32, 32), (32, 32))
def test_arbitrary_op_chain_ttnn(
    in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
):
    add = builder.add(in0, in1)
    exp = builder.exp(in2)
    return builder.multiply(add, exp)


if __name__ == "__main__":
    test_functions = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isfunction
    )

    for function_name, func in test_functions:
        if function_name.startswith("test_"):
            func()
