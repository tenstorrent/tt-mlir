# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s

import inspect
import os

from ttmlir.test_utils import (
    compile_as_mlir_module,
    translate_ttnn_to_flatbuffer,
    ttir_to_ttnn,
    translate_ttmetal_to_flatbuffer,
    ttir_to_ttmetal,
)
from ttmlir.ttir_builder import Operand, TTIRBuilder

system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")


@translate_ttmetal_to_flatbuffer(output_file_name="test_exp.ttm")
@ttir_to_ttmetal(
    output_file_name="test_exp.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((128, 128))
def test_exp_ttmetal(in0: Operand, builder: TTIRBuilder):
    return builder.exp(in0)


@translate_ttmetal_to_flatbuffer(output_file_name="test_add.ttm")
@ttir_to_ttmetal(
    output_file_name="test_add.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((64, 128), (64, 128))
def test_add_ttmetal(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.add(in0, in1)


@translate_ttmetal_to_flatbuffer(output_file_name="test_multiply.ttm")
@ttir_to_ttmetal(
    output_file_name="test_multiply.mlir", system_desc_path=f"{system_desc_path}"
)
@compile_as_mlir_module((64, 64), (64, 64))
def test_multiply_ttmetal(in0: Operand, in1: Operand, builder: TTIRBuilder):
    return builder.multiply(in0, in1)


@translate_ttmetal_to_flatbuffer(output_file_name="test_arbitrary_op_chain.ttm")
@ttir_to_ttmetal(
    output_file_name="test_arbitrary_op_chain.mlir",
    system_desc_path=f"{system_desc_path}",
)
@compile_as_mlir_module((32, 32), (32, 32), (32, 32))
def test_arbitrary_op_chain_ttmetal(
    in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
):
    add = builder.add(in0, in1)
    exp = builder.exp(in2)
    mul = builder.multiply(add, exp)
    in3 = builder.empty(builder.get_shape(mul))
    return builder.multiply(mul, in3)


if __name__ == "__main__":
    test_functions = inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isfunction
    )

    for function_name, func in test_functions:
        if function_name.startswith("test_"):
            func()
