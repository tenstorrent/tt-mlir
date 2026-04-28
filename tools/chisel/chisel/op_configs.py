# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-op configuration registry for chisel preOp/postOp dispatch.

Register a ChiselOpConfig to override default preOp/postOp behavior for specific
MLIR op types. Registrations at module level are applied on import.

Usage:
    register_op_config(MyOp, ChiselOpConfig(skip_pcc=True))
    register_op_config(MyOp, ChiselOpConfig(pre_op=my_pre, post_op=my_post))
"""
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from _ttmlir_runtime import runtime as tt_runtime
from ttmlir.dialects import func, ttcore, ttnn

from .context import ChiselContext
from .ops import get_op_inputs, get_op_outputs
from .utils import retrieve_torch_tensor

logger = logging.getLogger("chisel")


@dataclass
class ChiselOpConfig:
    """Per-op behavior overrides for chisel preOp/postOp dispatch.

    Attributes:
        skip:           Entirely bypass preOp and postOp for this op type.
                        No records are written and the tensor pool is not updated.
                        Use for ops whose IR/FB output counts disagree or whose
                        tensors are not retrievable (e.g. cache identifiers).
        skip_pcc:       Run isolation golden but skip numerical PCC comparison.
                        Useful for ops whose device output is intentionally undefined
                        (e.g. ttnn.empty produces uninitialized memory).
        skip_accum_pcc: Same as skip_pcc but for the accumulation golden check.
        pre_op:   Callable replacing the default preOp body after op iterator
                  advance. Signature: (binary, program_context, op_context) -> None.
        post_op:  Callable replacing the entire default postOp body.
                  Signature: (binary, program_context, op_context) -> None.
                  When set, skip_pcc and skip_accum_pcc are ignored (custom fn
                  owns PCC logic).
    """

    skip: bool = False
    skip_pcc: bool = False
    skip_accum_pcc: bool = False
    pre_op: Optional[Callable] = None
    post_op: Optional[Callable] = None


_OP_CONFIGS: Dict[Type, ChiselOpConfig] = {}


def register_op_config(op_type: Type, config: ChiselOpConfig) -> None:
    """Register a ChiselOpConfig for a specific MLIR op class."""
    _OP_CONFIGS[op_type] = config


def get_op_config(op: object) -> ChiselOpConfig:
    """Return the ChiselOpConfig for op's type, or a default config if not registered."""
    return _OP_CONFIGS.get(type(op), ChiselOpConfig())


def get_skipped_op_names() -> frozenset:
    """MLIR op names (e.g. 'ttcore.load_cached') for all ops registered with skip=True."""
    return frozenset(
        op_type.OPERATION_NAME
        for op_type, config in _OP_CONFIGS.items()
        if config.skip
    )


# ---------------------------------------------------------------------------
# load_cached handlers
# ---------------------------------------------------------------------------


def _load_cached_pre_op(binary, program_context, op_context) -> None:
    ctx = ChiselContext.get_instance()
    program = ctx.current_program
    binary_state = ctx.current_binary
    op = program.current_op
    asm_state = binary_state.ir_module.get_asm_state()

    callee_name = op.opview.callee.value
    callee_args = binary_state.ir_module.get_function_inputs(callee_name)
    parent_inputs = get_op_inputs(op)

    # Propagate parent golden values into global_tensor_pool keyed by the
    # callee's input arg SSA names.  preprogram for the callee sub-program
    # seeds from global_tensor_pool, so the callee gets correct accumulated
    # goldens instead of raw device tensors.
    for parent_inp, callee_arg in zip(parent_inputs, callee_args):
        parent_name = parent_inp.get_name(asm_state)
        callee_arg_name = callee_arg.get_name(asm_state)
        if parent_name in program.golden_tensor_pool:
            binary_state.global_tensor_pool[callee_arg_name] = (
                program.golden_tensor_pool[parent_name]
            )

    program.stashed_inputs = {}


def _load_cached_post_op(binary, program_context, op_context) -> None:
    from .checker import ChiselChecker

    ctx = ChiselContext.get_instance()
    program = ctx.current_program
    binary_state = ctx.current_binary
    op = program.current_op
    asm_state = binary_state.ir_module.get_asm_state()
    checker = ChiselChecker(ctx, op.name)

    op_outputs = get_op_outputs(op)
    if not op_outputs:
        program.stashed_inputs = None
        return

    output_refs = tt_runtime.get_op_output_refs(op_context, program_context)
    callee_name = op.opview.callee.value
    callee_returns = binary_state.ir_module.get_function_outputs(callee_name)

    for mlir_output, output_ref, callee_ret in zip(
        op_outputs, output_refs, callee_returns, strict=True
    ):
        result_name = mlir_output.get_name(asm_state)
        callee_ret_name = callee_ret.get_name(asm_state)

        golden = binary_state.global_tensor_pool.get(callee_ret_name)
        device = retrieve_torch_tensor(
            program_context, output_ref,
            checker=checker, slot=result_name, check="retrieve_output",
        )

        if golden is not None and device is not None:
            checker.check_golden_vs_runtime_tensor(result_name, golden, device)

        # Seed parent pool with callee golden so downstream ops get correct
        # accumulated golden inputs rather than device values.
        if golden is not None:
            program.golden_tensor_pool[result_name] = golden

    program.stashed_inputs = None


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

# ttnn.empty produces an uninitialized device tensor — PCC comparison is
# meaningless for both isolation and accumulation checks.
register_op_config(ttnn.EmptyOp, ChiselOpConfig(skip_pcc=True, skip_accum_pcc=True))

# ttcore.load_cached: golden propagation via BinaryState.global_tensor_pool.
# pre_op seeds callee input goldens; post_op maps callee return goldens back
# to the parent program pool and compares against device outputs.
register_op_config(ttcore.LoadCachedOp, ChiselOpConfig(
    pre_op=_load_cached_pre_op,
    post_op=_load_cached_post_op,
))

# ttnn.generic: IR output count = 0 but FB output count = 1.
register_op_config(ttnn.GenericOp, ChiselOpConfig(skip=True))

# func.call (cpu-hoisted): runtime fires callbacks for these via the flatbuffer
# CpuOp. Golden simulation is not possible — see _cpu_hoisted_post_op in
# callbacks.py, which registers the handler and seeds the golden pool with
# the device output.

# ttnn.paged_update_cache / ttnn.fill_cache: in-place cache writes with no
# retrievable output tensors.
register_op_config(ttnn.PagedUpdateCacheOp, ChiselOpConfig(skip=True))
register_op_config(ttnn.FillCacheOp, ChiselOpConfig(skip=True))

# scale/zero_point operands are stored as FB attribute structs, not TensorRefs,
# so IR and FB input counts disagree.
register_op_config(ttnn.QuantizeOp, ChiselOpConfig(skip=True))
register_op_config(ttnn.DequantizeOp, ChiselOpConfig(skip=True))
register_op_config(ttnn.RequantizeOp, ChiselOpConfig(skip=True))
