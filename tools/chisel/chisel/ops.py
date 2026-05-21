# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLIR operation utilities: IRModule wrapper, tensor operand extraction, and
chisel-specific op classification (non-executable / in-place).
"""
from typing import NewType, Optional, Tuple

from ttmlir.dialects import func, ttcore, ttnn
from ttmlir.ir import (
    AsmState,
    BlockArgument,
    Context,
    Module,
    OpOperandList,
    OpResult,
    Operation,
    Value,
    WalkOrder,
    WalkResult,
)


# MLIR SSA value name as printed in the IR (e.g. "%0", "%arg1"). Produced by
# `Value.get_name(asm_state)` and used to key per-op input/output tensors.
SSAName = NewType("SSAName", str)

# Operand role name on an OpView (e.g. "lhs", "rhs", "input"). Sourced from
# `OpView.OPERAND_NAMES` and CHISEL_INPLACE_OPS; used to dispatch goldens by
# their declared keyword arguments.
RoleName = NewType("RoleName", str)


# Op class -> operand role names mutated via `Arg<..., [MemWrite]>` in ODS.
# Goldens return SSA results first, then one tensor per *provided* memwrite
# operand (absent Optional operands are skipped - see get_flat_inplace_vals).
# Hand-maintained;
# TODO(ndrakulic, #8385): derive from ODS via python_op_schema_codegen.py.
_CHISEL_INPLACE_OPS: dict[type, tuple[str, ...]] = {
    ttnn.UpdateCacheOp: ("cache",),
    ttnn.PagedUpdateCacheOp: ("cache",),
    ttnn.FillCacheOp: ("cache",),
    ttnn.PagedFillCacheOp: ("cache",),
    ttnn.WriteTensorOp: ("device_tensor",),
    ttnn.DumpTensorOp: (),
    ttnn.DeallocateOp: (),
    ttnn.BatchNormTrainingOp: ("running_mean", "running_var"),
    ttnn.PointToPointOp: ("optional_output_tensor",),
}


def get_inplace_operands(op_class: type) -> tuple[RoleName, ...]:
    return _CHISEL_INPLACE_OPS.get(op_class, ())


def is_tensor_value(val: Value) -> bool:
    """True if `val` is a tensor-like MLIR Value (has shape and element_type)."""
    return hasattr(val.type, "shape") and hasattr(val.type, "element_type")


def get_op_outputs(op: Operation) -> list[OpResult]:
    """Extract output tensors (results with shape and element_type) from a MLIR operation."""
    return [result for result in op.results if is_tensor_value(result)]


def get_op_inputs(op: Operation) -> list[Value]:
    """Extract input tensors (operands with shape and element_type) from a MLIR operation."""
    return [operand for operand in op.operands if is_tensor_value(operand)]


def get_flat_inplace_vals(op: Operation) -> list[tuple[str, Value]]:
    """Return (role, value) pairs for in-place operands present on `op`.

    Absent Optional operands are skipped; OpOperandList is expanded.
    """
    vals: list[tuple[str, Value]] = []
    for role in get_inplace_operands(type(op)):
        accessor = getattr(op, role, None)
        if accessor is None:
            continue
        if isinstance(accessor, OpOperandList):
            vals.extend((role, v) for v in accessor)
        else:
            vals.append((role, accessor))
    return vals


class IRModule:
    """
    Wrapper around a MLIR Module with function lookup and operation traversal.

    Accepts a MLIR source string, parses it internally, and provides cached
    access to functions, operations, and assembly state.
    """

    def __init__(
        self,
        mlir_source: str,
        functions: list[str],
    ):
        self.context = Context()
        self.context.allow_unregistered_dialects = True
        self.module: Module = Module.parse(mlir_source, self.context)

        self._functions: dict[str, func.FuncOp] = {
            name: self._find_function(name) for name in functions
        }
        self._function_ops: dict[str, list[Operation]] = {}
        self._function_outputs: dict[str, list[Value]] = {}
        for name in functions:
            ops, outputs = self._extract_function_ops(name)
            self._function_ops[name] = ops
            self._function_outputs[name] = outputs
        self._asm_state = AsmState(self.module.operation)

    def get_asm_state(self) -> AsmState:
        """Module-wide AsmState (speeds up get_name calls)."""
        return self._asm_state

    def get_mesh_shape(self) -> Tuple[int, ...]:
        """Mesh shape from the module's `ttcore.meshes` attribute.

        Returns `(1, 1)` when the attribute is absent (single-chip programs).
        When multiple meshes are declared, the first one is used.
        """
        for named_attr in self.module.operation.attributes:
            if named_attr.name != "ttcore.meshes":
                continue
            meshes = ttcore.ir.MeshesAttr.maybe_downcast(named_attr.attr)
            if meshes is None or not meshes.meshes:
                continue
            return tuple(int(d) for d in meshes.meshes[0].shape)
        return (1, 1)

    def get_function(self, function_name: str) -> func.FuncOp:
        """The func.FuncOp for the given function."""
        return self._functions[function_name]

    def get_function_inputs(self, function_name: str) -> list[BlockArgument]:
        """Input arguments of the given function."""
        return self._functions[function_name].arguments

    def get_function_outputs(self, function_name: str) -> list[Value]:
        """Output values of the given function (operands of func.return)."""
        return self._function_outputs[function_name]

    def get_function_ops(self, function_name: str) -> list[Operation]:
        """Operations in the given function body."""
        return self._function_ops[function_name]

    def _extract_function_ops(self, name: str) -> tuple[list[Operation], list[Value]]:
        assert name in self._functions
        func_op = self._functions[name]
        ops: list[Operation] = []
        outputs: list[Value] = []

        def _visitor(op: Operation) -> WalkResult:
            # Python bindings only expose walk() on Operation; skip the FuncOp
            # itself to match C++ entry.getBody().walk() in FuncOpToProgram.h.
            if op == func_op.operation:
                return WalkResult.ADVANCE
            if op.name == "func.return":
                outputs.extend(op.operands)
                return WalkResult.ADVANCE
            ops.append(op)
            return WalkResult.ADVANCE

        func_op.operation.walk(_visitor, walk_order=WalkOrder.PRE_ORDER)
        return ops, outputs

    def _find_function(self, name: str) -> func.FuncOp:
        result: Optional[func.FuncOp] = None

        def _visitor(op: Operation) -> WalkResult:
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
