# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden execution for a single TTNN op via CHISEL_GOLDEN_MAPPINGS.

Golden interface: fn(op, inputs: Dict[RoleName, GoldenMapTensor]) -> GoldenMapTensor | tuple.
The executor normalizes the result to a list with one entry per SSA result
followed by one entry per provided in-place operand.
"""
import logging
from typing import Dict, List

from ttmlir.ir import OpOperandList, Operation, Value

from golden import get_chisel_golden_function, GoldenMapTensor

from .exceptions import GoldenNotImplementedError
from .ops import (
    RoleName,
    SSAName,
    get_inplace_vals,
    get_op_outputs,
    is_tensor_value,
)

logger = logging.getLogger("chisel")


def build_role_keyed_inputs(
    op: Operation, ssa_inputs: Dict[SSAName, GoldenMapTensor], asm_state
) -> Dict[RoleName, object]:
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

    role_inputs: Dict[RoleName, object] = {}
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
    golden_inputs: Dict[RoleName, object],
) -> List[GoldenMapTensor]:
    """Run the registered golden for `op` and return its outputs as a list.

    Raises GoldenNotImplementedError if no golden is registered for `op`, or
    TypeError if the golden returns an unsupported result type.
    """
    golden_fn = get_chisel_golden_function(type(op))
    if golden_fn is None:
        raise GoldenNotImplementedError(op)

    result = golden_fn(op, golden_inputs)

    if isinstance(result, GoldenMapTensor):
        tensors: List[GoldenMapTensor] = [result]
    elif isinstance(result, (list, tuple)) and all(
        isinstance(t, GoldenMapTensor) for t in result
    ):
        tensors = list(result)
    else:
        raise TypeError(
            f"Golden for {type(op).__name__} returned unsupported type "
            f"{type(result).__name__}"
        )

    ssa_count = len(get_op_outputs(op))
    inplace_vals = get_inplace_vals(op)
    expected = ssa_count + len(inplace_vals)
    assert len(tensors) == expected, (
        f"Golden for {type(op).__name__} returned {len(tensors)} tensor(s) "
        f"but op has {ssa_count} SSA output(s) + {len(inplace_vals)} mutated "
        f"operand(s); fix CHISEL_GOLDEN_MAPPINGS to return SSA outputs first, "
        f"then one tensor per provided in-place operand"
    )
    return tensors


def execute_golden_with_ssa_inputs(
    op: Operation,
    ssa_inputs: Dict[SSAName, GoldenMapTensor],
    asm_state,
) -> List[GoldenMapTensor]:
    """Re-key SSA-keyed inputs by role and run the registered golden for `op`."""
    role_inputs = build_role_keyed_inputs(op, ssa_inputs, asm_state)
    return execute_golden(op, role_inputs)
