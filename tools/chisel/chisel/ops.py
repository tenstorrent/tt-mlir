# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLIR operation utilities: IRModule wrapper and tensor operand extraction.
"""
from functools import cache
from typing import Dict, List, Optional

from ttmlir.dialects import func
from ttmlir.ir import (
    AsmState,
    Context,
    Module,
    Operation,
    WalkOrder,
    WalkResult,
    BlockArgument,
)


def get_op_outputs(op: Operation) -> list:
    """Extract output tensors (results with shape and element_type) from an MLIR operation."""
    outputs = []
    for result in op.results:
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(result)
    return outputs


def get_op_inputs(op: Operation) -> list:
    """Extract input tensors (operands with shape and element_type) from an MLIR operation."""
    inputs = []
    for operand in op.operands:
        if hasattr(operand.type, "shape") and hasattr(operand.type, "element_type"):
            inputs.append(operand)
    return inputs


class IRModule:
    """
    Wrapper around an MLIR Module with function lookup and operation traversal.

    Accepts an MLIR source string, parses it internally, and provides cached
    access to functions, operations, and assembly state.
    """

    def __init__(
        self,
        mlir_source: str,
        functions: List[str],
        current_function_name: Optional[str] = None,
        ignored_ops: List[str] = [],
    ):
        self.context = Context()
        self.context.allow_unregistered_dialects = True
        self.module: Module = Module.parse(mlir_source, self.context)
        self.ignored_ops: List[str] = ignored_ops

        self._functions: Dict[str, Operation] = {
            name: self._find_function(name) for name in functions
        }
        self._function_ops: Dict[str, List[Operation]] = {
            name: self._extract_function_ops(name) for name in functions
        }
        self._asm_state: Dict[str, AsmState] = {
            name: AsmState(self._functions[name]) for name in functions
        }

        if current_function_name is not None:
            self.current_function_name = current_function_name
        else:
            self.current_function_name = functions[0]

    def get_asm_state(self) -> AsmState:
        """AsmState for the current function (speeds up get_name calls)."""
        return self._asm_state[self.current_function_name]

    def get_function(self) -> Operation:
        """The current func.FuncOp."""
        return self._functions[self.current_function_name]

    def get_function_inputs(self) -> List[BlockArgument]:
        """Input arguments of the current function."""
        return self._functions[self.current_function_name].arguments

    def get_function_ops(self) -> List[Operation]:
        """Operations in the current function body (respecting ignored_ops)."""
        return self._function_ops[self.current_function_name]

    def _extract_function_ops(self, name: str) -> List[Operation]:
        assert name in self._functions
        func_op = self._functions[name]
        ops = []

        def _visitor(op):
            # Skip the FuncOp itself — Python bindings only expose walk() on
            # Operation, not Region, so we walk the FuncOp and skip it to match
            # the C++ entry.getBody().walk() in FuncOpToProgram.h.
            if op == func_op.operation:
                return WalkResult.ADVANCE
            if op.name == "func.return" or op.name in self.ignored_ops:
                return WalkResult.ADVANCE
            ops.append(op)
            return WalkResult.ADVANCE

        func_op.operation.walk(_visitor, walk_order=WalkOrder.PRE_ORDER)
        return ops

    def _find_function(self, name: str) -> Operation:
        result = None

        def _visitor(op):
            nonlocal result
            opview = op.opview
            if isinstance(opview, func.FuncOp):
                if opview.name.value == name:
                    result = opview
                    return WalkResult.INTERRUPT
                return WalkResult.SKIP
            return WalkResult.ADVANCE

        self.module.operation.walk(_visitor, walk_order=WalkOrder.PRE_ORDER)
        if result is None:
            raise ValueError(f"Function {name} not found in module")
        return result
