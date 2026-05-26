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
    OpResult,
    Operation,
    Value,
    WalkOrder,
    WalkResult,
)
from ttmlir.util import get_write_effect_operand_indices


# MLIR SSA value name as printed in the IR (e.g. "%0", "%arg1"). Produced by
# `Value.get_name(asm_state)` and used to key per-op input/output tensors.
SSAName = NewType("SSAName", str)

# Operand role name on an OpView (e.g. "lhs", "rhs", "input"). Sourced from
# `OpView.OPERAND_NAMES`; used to dispatch goldens by their declared keyword
# arguments and to label in-place mutation records.
RoleName = NewType("RoleName", str)


def is_tensor_value(val: Value) -> bool:
    """True if `val` is a tensor-like MLIR Value (has shape and element_type)."""
    return hasattr(val.type, "shape") and hasattr(val.type, "element_type")


def get_op_outputs(op: Operation) -> list[OpResult]:
    """Extract output tensors (results with shape and element_type) from a MLIR operation."""
    return [result for result in op.results if is_tensor_value(result)]


def get_op_inputs(op: Operation) -> list[Value]:
    """Extract input tensors (operands with shape and element_type) from a MLIR operation."""
    return [operand for operand in op.operands if is_tensor_value(operand)]


def get_inplace_input_refs(op, input_refs, asm_state) -> list:
    """Pair each in-place mutated tensor operand with its TensorRef.

    Returns a list of `(role, ssa, tensor_ref)`. `input_refs` is the flat
    list of tensor-typed operand refs from the runtime (same order as
    `get_op_inputs(op)`); in-place operands are a subset of those, so we
    walk both in parallel and emit a triple when an SSA name matches one
    of the in-place values reported by `get_flat_inplace_vals`.
    """
    inplace_pairs = get_flat_inplace_vals(op)
    if not inplace_pairs:
        return []
    role_by_ssa = {val.get_name(asm_state): role for role, val in inplace_pairs}
    out = []
    for mlir_in, ref in zip(get_op_inputs(op), input_refs):
        ssa = mlir_in.get_name(asm_state)
        role = role_by_ssa.get(ssa)
        if role is None:
            continue
        out.append((role, ssa, ref))
    return out


def get_flat_inplace_vals(op) -> list[tuple[RoleName, Value]]:
    """Return (role, value) pairs for tensor operands `op` declares MemWrite on.

    Driven by MemoryEffectOpInterface via
    `ttmlir.util.get_write_effect_operand_indices`. Returns [] when:
      - the op doesn't implement MemoryEffectOpInterface (effects unknown),
      - the op writes to no operand, or
      - all write-effect operands are non-tensor (e.g. device handles).
    The C++ side flattens variadic operands, so OpOperandList expansion is
    handled there.

    Accepts either an MLIR `Operation` or an `OpView`. The C++ binding takes
    `MlirOperation`, so we normalize via `op.operation` when present.
    """
    mlir_op = getattr(op, "operation", op)
    indices = get_write_effect_operand_indices(mlir_op)
    if not indices:
        return []
    operands = list(mlir_op.operands)
    # OPERAND_NAMES lives on the OpView class. If `op` is an Operation, its
    # `.opview` gives us the typed view; if it already is an OpView, use it.
    opview = op.opview if hasattr(op, "opview") else op
    names = getattr(type(opview), "OPERAND_NAMES", None) or []
    out: list[tuple[RoleName, Value]] = []
    for i in indices:
        if i >= len(operands):
            continue
        val = operands[i]
        if not is_tensor_value(val):
            continue
        role = RoleName(names[i] if i < len(names) else f"operand[{i}]")
        out.append((role, val))
    return out


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
        Raises if the module declares more than one mesh; chisel currently
        assumes a single mesh per module.
        """
        for named_attr in self.module.operation.attributes:
            if named_attr.name != "ttcore.meshes":
                continue
            meshes = ttcore.ir.MeshesAttr.maybe_downcast(named_attr.attr)
            if meshes is None or not meshes.meshes:
                continue
            if len(meshes.meshes) > 1:
                raise ValueError(
                    f"chisel does not support modules with more than one mesh; "
                    f"got {len(meshes.meshes)} meshes in `ttcore.meshes`"
                )
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
