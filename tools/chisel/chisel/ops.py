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


@cache
def get_op_outputs(op: Operation) -> list:
    """Extract output tensors (results with shape and element_type) from an MLIR operation."""
    outputs = []
    for result in op.results:
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(result)
    return outputs


@cache
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
        current_function_name: str | None = None,
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
        ops = []
        for region in self._functions[name].regions:
            for block in region.blocks:
                for op in block.operations:
                    if op.name in self.ignored_ops:
                        continue
                    ops.append(op)
        return ops

    def _find_function(self, name: str) -> Operation:
        for op in self._walk(self.module.operation):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                return op
        raise ValueError(f"Function {name} not found in module")

    def _walk(
        self, op: Operation, walk_order: WalkOrder = WalkOrder.POST_ORDER
    ) -> List[Operation]:
        ops = []

        def _walk_ops(op):
            nonlocal ops
            ops.append(op.opview)
            return WalkResult.ADVANCE

        op.operation.walk(_walk_ops, walk_order=walk_order)
        return ops
