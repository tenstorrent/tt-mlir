# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Dict

from ttmlir.ir import Operation, WalkOrder, WalkResult, Context, Module
from ttmlir.dialects import func

from .enums import ExecutionType
from ..utils.location import hash_location

def get_op_outputs(op: Operation):
    outputs = []
    for result in op.results:
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(result)
    return outputs

class IRModule:
    def __init__(
        self,
        mlir_text: str,
        context: Context,
        execution_type: ExecutionType,
        main_function_name: str,
        ignored_ops: List[str] = [],
    ):
        """
        Create an IRModule from MLIR text

        Args:
            mlir_text (str): MLIR text
            context (Context): Context
            execution_type (ExecutionType): Execution type
            main_function_name (str): Main function name
            ignored_ops (List[str], optional): Ops in MLIR to ignore.
        """
        self.mlir_module = Module.parse(mlir_text, context)
        self.context = context
        self.execution_type = execution_type
        self.main_function_name = main_function_name
        self._functions: Dict[str, Operation] = {}
        self._function_ops: Dict[str, List[Operation]] = {}

        self.ignored_ops = ignored_ops

        if main_function_name is not None:
            self._functions[main_function_name] = self.add_function(main_function_name)

    def add_function(self, name: str):
        if name in self._functions:
            print(f"Function with name {name} already exists")
            return self._functions[name]
        for op in self._dfs(self.mlir_module.operation):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                return op

    def get_function(self, name: str | None):
        if name is None:
            name = self.main_function_name
        return self._functions[name]

    def get_function_inputs(self, name: str | None) -> List[Operation]:
        if name is None:
            name = self.main_function_name
        assert name in self._functions
        return self._functions[name].arguments

    def get_function_ops(self, name: str | None) -> List[Operation]:
        if name is None:
            name = self.main_function_name
        if name in self._function_ops:
            return self._function_ops[name]
        op = self._functions[name]
        assert op is not None

        ops = []
        for region in op.regions:
            for block in region.blocks:
                for op in block.operations:
                    if op.name in self.ignored_ops:
                        continue
                    ops.append(op)

        self._function_ops[name] = ops
        return ops

    def _dfs(self, op: Operation, walk_order: WalkOrder = WalkOrder.POST_ORDER):
        assert op is not None
        ops = []

        def _walk_ops(op):
            nonlocal ops
            if op.name not in self.ignored_ops:
                ops.append(op.opview)
            return WalkResult.ADVANCE

        op.operation.walk(_walk_ops, walk_order=walk_order)
        return ops
