# tools/chisel/tests/test_executor.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


def test_build_golden_args_unary():
    """Test that _build_golden_args correctly builds args for a unary golden function."""
    from chisel.executor import _build_golden_args
    from golden import GoldenMapTensor

    def mock_golden(input_tensor: GoldenMapTensor, output_type_mlir):
        pass

    input_tensor = GoldenMapTensor({0: torch.randn(4, 4)}, (1, 1))
    output_type = "mock_type"

    class MockOp:
        attributes = {}

    args = _build_golden_args(mock_golden, [input_tensor], MockOp(), output_type)
    assert len(args) == 2
    assert args[0] is input_tensor
    assert args[1] == "mock_type"


def test_build_golden_args_binary():
    from chisel.executor import _build_golden_args
    from golden import GoldenMapTensor

    def mock_golden(lhs: GoldenMapTensor, rhs: GoldenMapTensor, output_type_mlir):
        pass

    t1 = GoldenMapTensor({0: torch.randn(4, 4)}, (1, 1))
    t2 = GoldenMapTensor({0: torch.randn(4, 4)}, (1, 1))

    class MockOp:
        attributes = {}

    args = _build_golden_args(mock_golden, [t1, t2], MockOp(), "type")
    assert len(args) == 3
    assert args[0] is t1
    assert args[1] is t2
    assert args[2] == "type"


def test_build_golden_args_with_attrs():
    from chisel.executor import _build_golden_args
    from golden import GoldenMapTensor

    def mock_golden(input_tensor: GoldenMapTensor, transpose_a_attr, output_type_mlir):
        pass

    t1 = GoldenMapTensor({0: torch.randn(4, 4)}, (1, 1))
    mock_attr = "mock_bool_attr"

    class MockOp:
        attributes = {"transpose_a": mock_attr}

    args = _build_golden_args(mock_golden, [t1], MockOp(), "type")
    assert len(args) == 3
    assert args[0] is t1
    assert args[1] == mock_attr
    assert args[2] == "type"


def test_build_golden_args_variadic():
    from chisel.executor import _build_golden_args
    from golden import GoldenMapTensor
    from typing import List

    def mock_golden(input_tensors: List[GoldenMapTensor], dim_attr, output_type_mlir):
        pass

    t1 = GoldenMapTensor({0: torch.randn(4, 4)}, (1, 1))
    t2 = GoldenMapTensor({0: torch.randn(4, 4)}, (1, 1))

    class MockOp:
        attributes = {"dim": 1}

    args = _build_golden_args(mock_golden, [t1, t2], MockOp(), "type")
    assert len(args) == 3
    assert isinstance(args[0], list)
    assert len(args[0]) == 2
    assert args[1] == 1
    assert args[2] == "type"


def test_execute_golden_unmapped_op_raises():
    """Verify RuntimeError for ops not in GOLDEN_MAPPINGS."""
    from chisel.ops import IRModule
    from chisel.executor import execute_golden

    # func.return is not in GOLDEN_MAPPINGS
    MODULE = """
    module {
      func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
        %0 = "test.abs"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
        return %0 : tensor<4x4xf32>
      }
    }
    """
    ir = IRModule(mlir_source=MODULE, functions=["main"])
    return_op = ir.get_function_ops()[-1]  # func.return

    with pytest.raises((RuntimeError, AssertionError)):
        execute_golden(return_op, ir, {})
