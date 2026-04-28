# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Golden execution for a single TTNN op via CHISEL_GOLDEN_MAPPINGS.

Each golden function has the interface:
    fn(op: Operation, inputs: Dict[str, GoldenMapTensor], asm_state: AsmState) -> GoldenMapTensor

The executor passes GoldenMapTensor inputs directly to the golden function and
normalises the result to a tuple of GoldenMapTensor with one entry per op
output. Single-tensor goldens on multi-output ops (e.g. sort/topk return only
values) are broadcast to every output slot so downstream consumers of any
output find a value.

For single-device programs each GoldenMapTensor wraps exactly one shard
({0: tensor}, (1,1)).  For multi-device programs there is one shard per device.
"""
import logging
import traceback
from typing import Dict, Iterable, Optional, Tuple

import torch
from ttmlir.ir import Operation

from golden import GoldenMapTensor, get_chisel_golden_function

from .ops import IRModule, get_op_inputs, get_op_outputs
from .tensors import TensorPool

logger = logging.getLogger("chisel")


def _record_many(checker, slots: Iterable[str], check: str, status: str, **extra) -> None:
    if checker is None:
        return
    for slot in slots:
        checker.record(slot, check, status, **extra)


def execute_golden(
    op: Operation,
    ir_module: IRModule,
    inputs: Dict[str, GoldenMapTensor],
    *,
    checker=None,
    slots: Optional[Iterable[str]] = None,
    check: str = "golden",
) -> Optional[Tuple[GoldenMapTensor, ...]]:
    """
    Execute a TTNN op on CPU via CHISEL_GOLDEN_MAPPINGS.

    Args:
        op: The MLIR operation to execute.
        ir_module: The IRModule containing the operation (for SSA name resolution).
        inputs: Dict mapping SSA names to GoldenMapTensor (device inputs copied to host).
        checker: Optional ChiselChecker — on failure records "skipped"/"error"
                 against each entry of `slots`. Pass None to suppress recording.
        slots: Per-output slot names used when recording. Required if `checker`
               is provided; ignored otherwise.
        check: Check label used for the recorded entry.

    Returns:
        Tuple[GoldenMapTensor, ...] — one GoldenMapTensor per op output. Single-tensor
        goldens on multi-output ops are broadcast to every slot.
        None if the golden is missing or raised.
    """
    slot_list = list(slots) if slots is not None else []
    golden_fn = get_chisel_golden_function(type(op))
    if golden_fn is None:
        msg = f"No golden implementation for {type(op).__name__}"
        logger.debug(f"{op.name}: {check} skipped ({msg})")
        _record_many(checker, slot_list, check, "skipped")
        return None

    op_inputs = get_op_inputs(op)
    asm_state = ir_module.get_asm_state()
    try:
        golden_inputs: Dict[str, GoldenMapTensor] = {
            inp.get_name(asm_state): inputs[inp.get_name(asm_state)]
            for inp in op_inputs
        }
    except KeyError as e:
        msg = f"missing input for golden: {e}"
        logger.debug(f"{op.name}: {check} skipped ({msg})")
        _record_many(checker, slot_list, check, "skipped")
        return None

    try:
        result = golden_fn(op, golden_inputs, asm_state)
    except Exception:
        tb = traceback.format_exc()
        logger.error(f"{op.name}: {check} error\n{tb}")
        _record_many(checker, slot_list, check, "error", traceback=tb)
        return None

    # Normalise result to a tuple of GoldenMapTensor.
    if isinstance(result, GoldenMapTensor):
        tensors: Tuple[GoldenMapTensor, ...] = (result,)
    elif isinstance(result, torch.Tensor):
        # Golden functions should always return GoldenMapTensor — a bare
        # torch.Tensor means the golden implementation is incorrect.
        logger.error(
            f"{op.name}: golden for {type(op).__name__} returned torch.Tensor "
            f"instead of GoldenMapTensor — fix the golden implementation"
        )
        tensors = (GoldenMapTensor({0: result}, (1, 1)),)
    elif isinstance(result, (list, tuple)):
        tensors = tuple(
            t if isinstance(t, GoldenMapTensor) else GoldenMapTensor({0: t}, (1, 1))
            for t in result
        )
    else:
        msg = (
            f"Golden for {type(op).__name__} returned unsupported type "
            f"{type(result).__name__}"
        )
        logger.error(f"{op.name}: {check} error ({msg})")
        _record_many(checker, slot_list, check, "error", traceback=msg)
        return None

    num_outputs = len(get_op_outputs(op))
    if len(tensors) == 1 and num_outputs > 1:
        tensors = tensors * num_outputs
    return tensors


def execute_golden_from_pool(
    op: Operation,
    ir_module: IRModule,
    tensor_pool: TensorPool,
    *,
    checker=None,
    slots: Optional[Iterable[str]] = None,
    check: str = "accum_golden",
) -> Optional[Tuple[GoldenMapTensor, ...]]:
    """
    Pool-aware golden execution.

    Pulls input GoldenMapTensors from tensor_pool by SSA name, calls execute_golden,
    stores each output GoldenMapTensor back in the pool under its SSA name, and
    returns the tuple (or None if the golden is missing/failed).

    ir_module is used to obtain asm_state for SSA name resolution of both
    inputs (pool lookup) and outputs (pool store).
    """
    slot_list = list(slots) if slots is not None else []
    asm_state = ir_module.get_asm_state()
    op_inputs = get_op_inputs(op)
    try:
        inputs = {
            inp.get_name(asm_state): tensor_pool[inp.get_name(asm_state)]
            for inp in op_inputs
        }
    except KeyError as e:
        logger.debug(f"{op.name}: {check} skipped (missing input in pool: {e})")
        _record_many(checker, slot_list, check, "skipped")
        return None

    # Inner call records its own failures under the same `check` label.
    result = execute_golden(
        op, ir_module, inputs, checker=checker, slots=slot_list, check=check,
    )
    if result is None:
        return None

    for out_val, res_tensor in zip(get_op_outputs(op), result, strict=True):
        tensor_pool[out_val.get_name(asm_state)] = res_tensor

    return result
