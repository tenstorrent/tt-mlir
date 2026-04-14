# tools/chisel/tests/test_context.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

SIMPLE_MODULE = """
module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "test.abs"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = "test.neg"(%0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
"""


@pytest.fixture(autouse=True)
def reset_singleton():
    from chisel.context import ChiselContext
    ChiselContext.reset_instance()
    yield
    ChiselContext.reset_instance()


def test_get_instance_before_init_raises():
    from chisel.context import ChiselContext
    with pytest.raises(RuntimeError, match="not initialized"):
        ChiselContext.get_instance()


def test_construction_sets_instance():
    from chisel.context import ChiselContext
    ctx = ChiselContext()
    assert ChiselContext.get_instance() is ctx


def test_singleton_returns_same_object():
    from chisel.context import ChiselContext
    ctx = ChiselContext()
    assert ChiselContext.get_instance() is ctx
    assert ChiselContext.get_instance() is ctx


def test_reset_clears_instance():
    from chisel.context import ChiselContext
    ChiselContext()
    ChiselContext.reset_instance()
    with pytest.raises(RuntimeError, match="not initialized"):
        ChiselContext.get_instance()


def test_op_iter_advances():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"], ignored_ops=["func.return"])
    ctx = ChiselContext()
    ctx.ir_module = ir
    ctx.op_iter = iter(ir.get_function_ops())
    op1 = next(ctx.op_iter)
    assert op1.name == "test.abs"
    op2 = next(ctx.op_iter)
    assert op2.name == "test.neg"


def test_op_iter_exhaustion():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule
    ir = IRModule(mlir_source=SIMPLE_MODULE, functions=["main"], ignored_ops=["func.return"])
    ctx = ChiselContext()
    ctx.ir_module = ir
    ctx.op_iter = iter(ir.get_function_ops())
    next(ctx.op_iter)
    next(ctx.op_iter)
    with pytest.raises(StopIteration):
        next(ctx.op_iter)


def test_stashed_inputs_lifecycle():
    from chisel.context import ChiselContext
    ctx = ChiselContext()
    assert ctx._stashed_inputs is None
    ctx._stashed_inputs = {"arg0": "tensor_data"}
    assert ctx._stashed_inputs["arg0"] == "tensor_data"
    ctx._stashed_inputs = None
    assert ctx._stashed_inputs is None


def test_current_op_lifecycle():
    from chisel.context import ChiselContext
    ctx = ChiselContext()
    assert ctx._current_op is None
    ctx._current_op = "mock_op"
    assert ctx._current_op == "mock_op"
