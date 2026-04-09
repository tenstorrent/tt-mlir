# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

SIMPLE_MODULE = """
module {
  func.func @main(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "test.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = "test.abs"(%0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
"""


def test_ir_module_creation():
    from chisel.ops import IRModule
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    assert ir.module is not None
    assert ir.current_function_name == "main"


def test_get_function_returns_func_op():
    from chisel.ops import IRModule
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    func_op = ir.get_function()
    assert func_op.name.value == "main"


def test_get_function_inputs():
    from chisel.ops import IRModule
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    inputs = ir.get_function_inputs()
    assert len(inputs) == 2


def test_get_function_ops_returns_correct_count():
    from chisel.ops import IRModule
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ops = ir.get_function_ops()
    # ttnn.add, ttnn.abs (func.return is skipped to match C++ serialization)
    assert len(ops) == 2


def test_get_function_ops_order():
    from chisel.ops import IRModule
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    ops = ir.get_function_ops()
    assert ops[0].name == "test.add"
    assert ops[1].name == "test.abs"


def test_ignored_ops():
    from chisel.ops import IRModule
    ir = IRModule(
        mlir_source=SIMPLE_MODULE,
        functions=["main"],
        ignored_ops=["test.abs"],
    )
    ops = ir.get_function_ops()
    assert len(ops) == 1
    assert all(op.name != "test.abs" for op in ops)
    assert ops[0].name == "test.add"


def test_get_asm_state():
    from chisel.ops import IRModule
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    asm_state = ir.get_asm_state()
    assert asm_state is not None


def test_get_op_inputs():
    from chisel.ops import IRModule, get_op_inputs
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    add_op = ir.get_function_ops()[0]  # ttnn.add
    inputs = get_op_inputs(add_op)
    assert len(inputs) == 2


def test_get_op_outputs():
    from chisel.ops import IRModule, get_op_outputs
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    add_op = ir.get_function_ops()[0]  # ttnn.add
    outputs = get_op_outputs(add_op)
    assert len(outputs) == 1


def test_get_op_inputs_for_unary():
    from chisel.ops import IRModule, get_op_inputs
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"])
    abs_op = ir.get_function_ops()[1]  # ttnn.abs
    inputs = get_op_inputs(abs_op)
    assert len(inputs) == 1


def test_missing_function_raises():
    from chisel.ops import IRModule
    with pytest.raises(ValueError, match="not found"):
        IRModule(mlir_source=SIMPLE_MODULE, functions=["nonexistent"])
