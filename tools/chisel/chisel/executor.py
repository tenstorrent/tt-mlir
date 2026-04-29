# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden execution for a single TTNN op via CHISEL_GOLDEN_MAPPINGS.

Each golden function has the interface:
    fn(op: Operation, inputs: Dict[str, GoldenMapTensor]) -> GoldenMapTensor

The caller is responsible for building the role-keyed input dict (operand role
name -> GoldenMapTensor / list[GoldenMapTensor] / None for an absent
Optional<Tensor>). The executor looks up the golden function for the op and
dispatches it on those inputs.
"""
import logging
import traceback
from typing import Dict, Iterable, Optional

from ttmlir.ir import Operation

from golden import get_chisel_golden_function, GoldenMapTensor

logger = logging.getLogger("chisel")


def execute_golden(
    op: Operation,
    inputs: Dict[str, object],
) -> Optional[GoldenMapTensor]:
    """
    Execute a TTNN op on CPU via CHISEL_GOLDEN_MAPPINGS.

    Args:
        op: The MLIR operation to execute.
        inputs: Dict keyed by operand role name (OPERAND_NAMES). Each value is
                a GoldenMapTensor for a single operand, a list[GoldenMapTensor]
                for a Variadic operand, or None for an absent Optional<Tensor>.

    Returns:
        GoldenMapTensor — the value the golden function returned.
        None if the golden is missing, raised, or returned an unsupported type.
    """
    golden_fn = get_chisel_golden_function(type(op))
    if golden_fn is None:
        msg = f"No golden implementation for {type(op).__name__}"
        logger.debug(f"{op.name}: golden skipped ({msg})")
        return None

    try:
        result = golden_fn(op, inputs)
    except Exception:
        tb = traceback.format_exc()
        logger.error(f"{op.name}: golden error\n{tb}")
        return None

    if not isinstance(result, GoldenMapTensor):
        msg = (
            f"Golden for {type(op).__name__} returned unsupported type "
            f"{type(result).__name__}"
        )
        logger.error(f"{op.name}: {check} error ({msg})")
        return None

    return result
