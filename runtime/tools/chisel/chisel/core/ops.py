# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from functools import cache
from pathlib import Path
from collections import defaultdict
from typing import List
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import pickle

from ttmlir.ir import Operation, WalkOrder, WalkResult, Context, Module
from ttmlir.dialects import func

from .enums import ExecutionType, Status
from ..utils.location import hash_location
import pdb

attr_type_set = set()


class IRModule:
    def __init__(
        self,
        mlir_text: str,
        context: Context,
        execution_type: ExecutionType,
        main_op_name: str,
    ):
        self.mlir_module = Module.parse(mlir_text, context)
        self.context = context
        self.execution_type = execution_type
        self.main_op_name = main_op_name
        self._main_op: Operation = self.set_main_op()

        self._main_body_ops: List[Operation] = []

    def set_main_op(self, name: str | None = None):
        if name is None:
            name = self.main_op_name
        for op in self._dfs(self.mlir_module.operation):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                self._main_op = op
                return op

    def get_function_inputs(self):
        assert self._main_op is not None
        return self._main_op.arguments

    def get_main_op(self):
        return self._main_op

    def main_body_ops(self) -> List[Operation]:
        if len(self._main_body_ops) != 0:
            return self._main_body_ops
        op = self._main_op
        assert op is not None
        ops = []
        for region in op.regions:
            for block in region.blocks:
                for op in tqdm(block.operations):
                    if op.name in ["ttir.empty", "ttnn.deallocate"]:
                        continue
                    ops.append(op)
        self._main_body_ops = ops
        return ops

    def _dfs(
        self, op: Operation | None = None, walk_order: WalkOrder = WalkOrder.POST_ORDER
    ):
        if op is None:
            op = self._main_op
            assert op is not None
        ops = []

        def _walk_ops(op):
            nonlocal ops
            if not op.name == "ttir.empty":
                ops.append(op.opview)
            return WalkResult.ADVANCE

        op.operation.walk(_walk_ops, walk_order=walk_order)
        return ops
