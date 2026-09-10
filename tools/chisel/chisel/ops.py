# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLIR operation utilities: IRModule wrapper, tensor operand extraction, and
chisel-specific op classification (non-executable / in-place).
"""
from typing import NewType, Tuple

from ttmlir.dialects import func, ttcore, ttnn
from ttmlir.ir import (
    AsmState,
    Block,
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
# arguments.
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


def get_inplace_vals(op) -> list[Value]:
    """Return tensor operands `op` declares MemWrite on, in flat operand order.

    Driven by MemoryEffectOpInterface via
    `ttmlir.util.get_write_effect_operand_indices`, which returns flat
    operand indices (variadics already expanded), or an empty list if the
    op doesn't implement the interface. Returns [] when:
      - the op doesn't implement MemoryEffectOpInterface (effects unknown),
      - the op writes to no operand, or
      - all write-effect operands are non-tensor (e.g. device handles).

    Accepts either an MLIR `Operation` or an `OpView`. The C++ binding takes
    `MlirOperation`, so we normalize via `op.operation` when present.
    """
    mlir_op = getattr(op, "operation", op)
    indices = get_write_effect_operand_indices(mlir_op)
    if not indices:
        return []
    operands = list(mlir_op.operands)
    return [operands[i] for i in indices if is_tensor_value(operands[i])]


def _while_region_program_name(
    function_name: str, loop_index: int, region_name: str
) -> str:
    """Name of the flatbuffer program a `ttnn.while` region is serialized into.

    Mirrors `createOp(FlatbufferObjectCache &, WhileOp, ...)` in
    lib/Target/TTNN/TTNNToFlatbuffer.cpp, which numbers loops by their
    pre-order position within the enclosing function.
    """
    return f"{function_name}_while_{loop_index}_{region_name}"


def _collect_while_ops(func_op: func.FuncOp) -> list[Operation]:
    """The function's `ttnn.while` ops, in the order the serializer numbers them."""
    while_ops: list[Operation] = []

    def _visitor(op: Operation) -> WalkResult:
        if op.name == ttnn.WhileOp.OPERATION_NAME:
            while_ops.append(op)
        return WalkResult.ADVANCE

    func_op.operation.walk(_visitor, walk_order=WalkOrder.PRE_ORDER)
    return while_ops


def _split_terminator(block: Block) -> tuple[list[Operation], list[Value]]:
    """Split `block` into its program operations and its outputs.

    Mirrors `blockOpsToProgram` in FuncOpToProgram.h: only the operations
    directly in the block become program operations - ops nested in a region
    belong to that region's own program - and the terminator's operands
    (`func.return` or `ttnn.yield`) are the program outputs.
    """
    ops = [op.operation for op in block.operations]
    terminator = ops.pop()
    return ops, list(terminator.operands)


class IRModule:
    """
    Wrapper around a MLIR Module with program lookup and operation traversal.

    Accepts a MLIR source string, parses it internally, and provides cached
    access to programs, operations, and assembly state.
    """

    def __init__(
        self,
        mlir_source: str,
        functions: list[str],
    ):
        self.context = Context()
        self.context.allow_unregistered_dialects = True
        self.module: Module = Module.parse(mlir_source, self.context)

        program_blocks = self._collect_program_blocks()
        self._blocks: dict[str, Block] = {}
        self._function_ops: dict[str, list[Operation]] = {}
        self._function_outputs: dict[str, list[Value]] = {}
        for name in functions:
            block = program_blocks.get(name)
            if block is None:
                raise ValueError(f"Function {name} not found in module")
            self._blocks[name] = block
            ops, outputs = _split_terminator(block)
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

    def get_function_inputs(self, function_name: str) -> list[BlockArgument]:
        """Input arguments of the given program."""
        return list(self._blocks[function_name].arguments)

    def get_function_outputs(self, function_name: str) -> list[Value]:
        """Output values of the given program (operands of its terminator)."""
        return self._function_outputs[function_name]

    def get_function_ops(self, function_name: str) -> list[Operation]:
        """Operations in the given program body."""
        return self._function_ops[function_name]

    def _collect_program_blocks(self) -> dict[str, Block]:
        """Map every flatbuffer program name to the block it is serialized from.

        A `func.func` becomes a program under its own symbol name. Each region
        of a `ttnn.while` becomes a program of its own, since the runtime
        executes it with a nested ProgramExecutor.
        """
        blocks: dict[str, Block] = {}

        def _visitor(op: Operation) -> WalkResult:
            opview = op.opview
            if not isinstance(opview, func.FuncOp) or not opview.body.blocks:
                return WalkResult.ADVANCE

            name = opview.name.value
            blocks[name] = opview.body.blocks[0]
            for loop_index, while_op in enumerate(_collect_while_ops(opview)):
                for region, region_name in zip(
                    while_op.regions, ttnn.WhileOp.REGION_NAMES
                ):
                    blocks[
                        _while_region_program_name(name, loop_index, region_name)
                    ] = region.blocks[0]
            return WalkResult.SKIP

        self.module.operation.walk(_visitor, walk_order=WalkOrder.PRE_ORDER)
        return blocks
