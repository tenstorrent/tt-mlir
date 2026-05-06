# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden execution for a single TTNN op via CHISEL_GOLDEN_MAPPINGS.

Golden interface: fn(op, inputs: Dict[str, GoldenMapTensor]) -> GoldenMapTensor | tuple.
The executor normalizes the result to a tuple with one entry per SSA result
followed by one entry per provided in-place operand.
"""
import logging
from typing import Dict, Iterable, Optional, Tuple

from ttmlir.ir import OpOperandList, Operation, Value

from golden import get_chisel_golden_function, GoldenMapTensor

from .exceptions import GoldenNotImplementedError
from .ops import (
    IRModule,
    get_inplace_operands,
    get_op_inputs,
    get_op_outputs,
    is_tensor_value,
)
from .tensors import TensorPool

logger = logging.getLogger("chisel")


def build_role_keyed_inputs(
    op: Operation, ssa_inputs: Dict[str, GoldenMapTensor], asm_state
) -> Dict[str, object]:
    """Re-key `ssa_inputs` by operand role name (per OpView's OPERAND_NAMES).

    Per role: `Value` -> tensor, `OpOperandList` -> list (Variadic), None ->
    absent Optional. Non-tensor operands (e.g. `device`) become None.
    """
    operand_names = getattr(type(op), "OPERAND_NAMES", None)
    assert operand_names is not None, (
        f"{type(op).__name__} is missing OPERAND_NAMES; "
        "OpView bindings must be codegened with operand names"
    )

    def lookup(val):
        if not is_tensor_value(val):
            return None
        return ssa_inputs[val.get_name(asm_state)]

    role_inputs: Dict[str, object] = {}
    for name in operand_names:
        accessor = getattr(op, name, None)
        if accessor is None:
            role_inputs[name] = None
        elif isinstance(accessor, OpOperandList):
            role_inputs[name] = [lookup(v) for v in accessor]
        elif isinstance(accessor, Value):
            role_inputs[name] = lookup(accessor)
        else:
            raise TypeError(
                f"{type(op).__name__}.{name} unexpected type "
                f"{type(accessor).__name__}"
            )
    return role_inputs


def execute_golden(
    op: Operation,
    golden_inputs: Dict[str, object],
) -> Optional[Tuple[GoldenMapTensor, ...]]:
    """Run the registered golden for `op` and return its outputs as a tuple.

    Returns None if the golden returns an unsupported type. Raises
    GoldenNotImplementedError if no golden is registered for `op`.
    """
    golden_fn = get_chisel_golden_function(type(op))
    if golden_fn is None:
        raise GoldenNotImplementedError(op)

    result = golden_fn(op, golden_inputs)

    if isinstance(result, GoldenMapTensor):
        tensors: Tuple[GoldenMapTensor, ...] = (result,)
    elif isinstance(result, (list, tuple)) and all(
        isinstance(t, GoldenMapTensor) for t in result
    ):
        tensors = tuple(result)
    else:
        msg = (
            f"Golden for {type(op).__name__} returned unsupported type "
            f"{type(result).__name__}"
        )
        logger.error(f"{op.name}: golden error ({msg})")
        return None

    ssa_count = len(get_op_outputs(op))
    inplace_vals = get_provided_inplace_vals(op, get_inplace_operands(type(op)))
    expected = ssa_count + len(inplace_vals)
    assert len(tensors) == expected, (
        f"Golden for {type(op).__name__} returned {len(tensors)} tensor(s) "
        f"but op has {ssa_count} SSA output(s) + {len(inplace_vals)} mutated "
        f"operand(s); fix CHISEL_GOLDEN_MAPPINGS to return SSA outputs first, "
        f"then one tensor per provided in-place operand"
    )
    return tensors


def get_provided_inplace_vals(
    op: Operation, roles: Iterable[str]
) -> list[tuple[str, Value]]:
    """Return (role, value) pairs for in-place operands present on `op`.

    Absent Optional operands are skipped; OpOperandList is expanded.
    """
    vals: list[tuple[str, Value]] = []
    for role in roles:
        accessor = getattr(op, role, None)
        if accessor is None:
            continue
        if isinstance(accessor, OpOperandList):
            vals.extend((role, v) for v in accessor)
        else:
            vals.append((role, accessor))
    return vals


def execute_golden_from_pool(
    op: Operation,
    ir_module: IRModule,
    tensor_pool: TensorPool,
) -> Optional[Tuple[GoldenMapTensor, ...]]:
    """
    Pool-aware golden execution.

    Pulls SSA-keyed inputs from tensor_pool, role-keys them, calls
    execute_golden, then stores each SSA-result tensor back into the pool
    under its SSA name. Returns the result tuple, or None if an input is
    missing from the pool or no golden is registered.
    """
    asm_state = ir_module.get_asm_state()
    op_inputs = get_op_inputs(op)
    try:
        ssa_inputs = {
            inp.get_name(asm_state): tensor_pool[inp.get_name(asm_state)]
            for inp in op_inputs
        }
    except KeyError as e:
        logger.debug(f"{op.name}: accum_golden skipped (missing input in pool: {e})")
        return None

    golden_inputs = build_role_keyed_inputs(op, ssa_inputs, asm_state)
    try:
        result = execute_golden(op, golden_inputs)
    except GoldenNotImplementedError:
        logger.debug(f"{op.name}: accum_golden skipped (no golden registered)")
        return None
    if result is None:
        return None

    ssa_outputs = get_op_outputs(op)
    for out_val, res_tensor in zip(ssa_outputs, result[: len(ssa_outputs)]):
        tensor_pool[out_val.get_name(asm_state)] = res_tensor

    return result
