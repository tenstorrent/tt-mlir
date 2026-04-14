# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLIR operation utilities: IRModule wrapper and tensor operand extraction.
"""
from functools import cache
from typing import Dict, List

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
    ):
        self.context = Context()
        self.context.allow_unregistered_dialects = True
        self.module: Module = Module.parse(mlir_source, self.context)

        self._functions: Dict[str, Operation] = {
            name: self._find_function(name) for name in functions
        }
        self._function_ops: Dict[str, List[Operation]] = {}
        self._function_outputs: Dict[str, list] = {}
        for name in functions:
            ops, outputs = self._extract_function_ops(name)
            self._function_ops[name] = ops
            self._function_outputs[name] = outputs
        self._asm_state: Dict[str, AsmState] = {
            name: AsmState(self._functions[name]) for name in functions
        }

    def get_asm_state(self, function_name: str) -> AsmState:
        """AsmState for the given function (speeds up get_name calls)."""
        return self._asm_state[function_name]

    def get_function(self, function_name: str) -> Operation:
        """The func.FuncOp for the given function."""
        return self._functions[function_name]

    def get_function_inputs(self, function_name: str) -> List[BlockArgument]:
        """Input arguments of the given function."""
        return self._functions[function_name].arguments

    def get_function_outputs(self, function_name: str) -> list:
        """Output values of the given function (operands of func.return)."""
        return self._function_outputs[function_name]

    def get_function_ops(self, function_name: str) -> List[Operation]:
        """Operations in the given function body."""
        return self._function_ops[function_name]

    def _extract_function_ops(self, name: str):
        assert name in self._functions
        func_op = self._functions[name]
        ops = []
        outputs = []

        def _visitor(op):
            # Skip the FuncOp itself — Python bindings only expose walk() on
            # Operation, not Region, so we walk the FuncOp and skip it to match
            # the C++ entry.getBody().walk() in FuncOpToProgram.h.
            if op == func_op.operation:
                return WalkResult.ADVANCE
            if op.name == "func.return":
                outputs.extend(op.operands)
                return WalkResult.ADVANCE
            ops.append(op)
            return WalkResult.ADVANCE

        func_op.operation.walk(_visitor, walk_order=WalkOrder.PRE_ORDER)
        return ops, outputs

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
