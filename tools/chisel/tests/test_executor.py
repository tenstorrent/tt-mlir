# tools/chisel/tests/test_executor.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


def test_execute_golden_calls_with_op_and_dict(monkeypatch):
    """Test that execute_golden passes (op, Dict[str, GoldenMapTensor], asm_state) to the golden fn."""
    from chisel.executor import execute_golden
    from chisel.ops import IRModule
    from golden import GoldenMapTensor, CHISEL_GOLDEN_MAPPINGS

    MODULE = """
    module {
      func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %0 = "test.abs"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
        return %0 : tensor<4x4xf32>
      }
    }
    """
    ir = IRModule(mlir_source=MODULE, functions=["main"])
    test_op = ir.get_function_ops("main")[0]
    op_type = type(test_op)

    captured = {}

    def mock_golden(op, inputs, asm_state):
        captured["op"] = op
        captured["inputs"] = inputs
        captured["asm_state"] = asm_state
        assert isinstance(inputs, dict)
        return GoldenMapTensor({0: torch.randn(4, 4)}, (1, 1))

    monkeypatch.setitem(CHISEL_GOLDEN_MAPPINGS, op_type, mock_golden)

    asm_state = ir.get_asm_state("main")
    input_name = test_op.operands[0].get_name(asm_state)
    inputs = {input_name: torch.randn(4, 4)}

    result = execute_golden(test_op, ir, "main", inputs)

    assert "op" in captured
    assert captured["op"] is test_op
    assert isinstance(captured["inputs"], dict)
    assert len(captured["inputs"]) == 1
    assert isinstance(result, torch.Tensor)


def test_execute_golden_unmapped_op_raises():
    """Verify RuntimeError for ops not in CHISEL_GOLDEN_MAPPINGS."""
    from chisel.ops import IRModule
    from chisel.executor import execute_golden

    MODULE = """
    module {
      func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %0 = "test.abs"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
        return %0 : tensor<4x4xf32>
      }
    }
    """
    ir = IRModule(mlir_source=MODULE, functions=["main"])
    test_op = ir.get_function_ops("main")[0]

    with pytest.raises(RuntimeError):
        execute_golden(test_op, ir, "main", {})
