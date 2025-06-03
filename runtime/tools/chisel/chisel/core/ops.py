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
from .tensors import TensorDescriptor, get_op_inputs, get_op_outputs
from ..utils.location import hash_location
from ..utils.mapping import resolve_dense_attr
from ..utils.mapping import ttir_dtype_maps, handle_attr_type
import pdb

attr_type_set = set()


class Op:
    def __init__(self, op: Operation, execution_type: ExecutionType):
        self.inputs: List[TensorDescriptor] = get_op_inputs(op, execution_type)
        self.outputs: List[TensorDescriptor] = get_op_outputs(op, execution_type)
        self.name = str(op.name)
        self.location = hash_location(op.location)
        self.asm = op.get_asm(enable_debug_info=True)
        self.execution_type = execution_type
        # self.ir_op = op
        if execution_type == ExecutionType.GOLDEN:
            self.args = self._process_args(op)
        else:
            self.args = None

    def _process_args(self, op: Operation):
        args = {}
        for attr in op.attributes:
            args[attr.name] = handle_attr_type[type(attr.attr)](attr.attr)
            attr_type_set.add(type(args[attr.name]))
        return args

    def __str__(self):
        return f"{self.name} {self.location} {self.asm}"


# @cache
def get_op(op: Operation, execution_type: ExecutionType) -> Op:
    return Op(op, execution_type)


class IRModule:
    def __init__(
        self,
        mlir_path: Path,
        context: Context,
        execution_type: ExecutionType,
        main_op_name: str,
    ):
        self.mlir_path = mlir_path
        self.mlir_module = Module.parse(mlir_path.read_text(), context)
        self.context = context
        self.execution_type = execution_type
        self.main_op_name = main_op_name
        self._main_op: Operation = self.set_main_op()

        self._main_body_ops = None
        pickle_path = mlir_path.with_suffix(f".{self.main_op_name}.pkl")
        if pickle_path.exists():
            self.load_main_body_ops(pickle_path)
        else:
            ops = self.main_body_ops()
            self.save_main_body_ops(pickle_path)

    def set_main_op(self, name: str | None = None):
        if name is None:
            name = self.main_op_name
        for op in self._dfs(self.mlir_module.operation):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                self._main_op = op
                return op

    def get_main_op(self):
        return self._main_op

    def save_main_body_ops(self, path: Path):
        ops = self.main_body_ops()
        with open(path, "wb") as f:
            pickle.dump(ops, f)

    def load_main_body_ops(self, path: Path):
        with open(path, "rb") as f:
            self._main_body_ops = pickle.load(f)

    def main_body_ops(self) -> List[Op]:
        if self._main_body_ops is not None:
            return self._main_body_ops
        op = self._main_op
        assert op is not None
        ops = []
        for region in op.regions:
            for block in region.blocks:
                for op in tqdm(block.operations):
                    if op.name in ["ttir.empty", "ttnn.deallocate"]:
                        continue
                    ops.append(get_op(op, self.execution_type))
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
