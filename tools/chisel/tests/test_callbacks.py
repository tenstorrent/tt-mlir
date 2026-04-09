# tools/chisel/tests/test_callbacks.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from unittest.mock import MagicMock, patch

SIMPLE_MODULE = """
module {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = "test.abs"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
"""


@pytest.fixture(autouse=True)
def reset_singleton():
    from chisel.context import ChiselContext
    ChiselContext.reset_instance()
    yield
    ChiselContext.reset_instance()


@pytest.fixture
def ctx_with_module():
    from chisel.context import ChiselContext
    from chisel.ops import IRModule
    ir = IRModule(
        mlir_source=SIMPLE_MODULE,
        functions=["main"],
        ignored_ops=["func.return"],
    )
    ctx = ChiselContext()
    ctx.ir_module = ir
    ctx.op_iter = iter(ir.get_function_ops())
    return ctx


def test_pre_op_advances_op_iter(ctx_with_module):
    from chisel.callbacks import chisel_pre_op_callback

    with patch("_ttmlir_runtime.runtime.get_op_input_refs", return_value=[]):
        chisel_pre_op_callback(MagicMock(), MagicMock(), MagicMock())

    ctx = ctx_with_module
    assert ctx._current_op is not None
    assert ctx._current_op.name == "test.abs"


def test_pre_op_stashes_inputs(ctx_with_module):
    from chisel.callbacks import chisel_pre_op_callback

    mock_ref = MagicMock()
    input_tensor = torch.randn(4, 4)

    with (
        patch("_ttmlir_runtime.runtime.get_op_input_refs", return_value=[mock_ref]),
        patch("_ttmlir_runtime.runtime.retrieve_tensor_from_pool", return_value=MagicMock()),
        patch("chisel.callbacks.get_torch_tensor", return_value=input_tensor),
    ):
        chisel_pre_op_callback(MagicMock(), MagicMock(), MagicMock())

    ctx = ctx_with_module
    assert ctx._stashed_inputs is not None
    assert len(ctx._stashed_inputs) == 1


def test_post_op_clears_stash(ctx_with_module):
    from chisel.callbacks import chisel_pre_op_callback, chisel_post_op_callback

    input_tensor = torch.randn(4, 4)
    golden_output = torch.abs(input_tensor)
    device_output = torch.abs(input_tensor) + 0.001

    mock_ref = MagicMock()

    # Pre-op: stash inputs
    with (
        patch("_ttmlir_runtime.runtime.get_op_input_refs", return_value=[mock_ref]),
        patch("_ttmlir_runtime.runtime.retrieve_tensor_from_pool", return_value=MagicMock()),
        patch("chisel.callbacks.get_torch_tensor", return_value=input_tensor),
    ):
        chisel_pre_op_callback(MagicMock(), MagicMock(), MagicMock())

    # Post-op: execute golden and compare
    with (
        patch("chisel.callbacks.execute_golden", return_value=golden_output),
        patch("_ttmlir_runtime.runtime.get_op_output_ref", return_value=mock_ref),
        patch("_ttmlir_runtime.runtime.retrieve_tensor_from_pool", return_value=MagicMock()),
        patch("chisel.callbacks.get_torch_tensor", return_value=device_output),
    ):
        chisel_post_op_callback(MagicMock(), MagicMock(), MagicMock())

    ctx = ctx_with_module
    assert ctx._stashed_inputs is None


def test_callback_without_context_raises():
    from chisel.callbacks import chisel_pre_op_callback
    with pytest.raises(RuntimeError, match="not initialized"):
        chisel_pre_op_callback(MagicMock(), MagicMock(), MagicMock())
