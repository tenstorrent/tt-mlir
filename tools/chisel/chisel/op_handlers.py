# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .ops import get_op_inputs, get_op_outputs
from .report import ChiselRecord, GoldenEvictedPayload
from .safety import chisel_safe
from .utils import (
    promote_golden,
    publish_to_session_pool,
    invalidate_device_cache,
)
from .validators import check_shape_dtype


@chisel_safe
def _deallocate_pre_op(ctx, config) -> None:
    """Evict each input SSA from both pools; emit one record per eviction."""
    golden_pool = ctx.golden_tensor_pool
    asm_state = ctx.asm_state
    op = ctx.op
    for inp in get_op_inputs(op):
        ssa = inp.get_name(asm_state)
        invalidate_device_cache(ctx, ssa)
        if golden_pool.pop(ssa, None) is None:
            continue

        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_evicted",
                ssa=ssa,
                payload=GoldenEvictedPayload(),
            )
        )


@chisel_safe
def _noop_post_op(ctx, config) -> None:
    pass


@chisel_safe
def _subprogram_pre_op(ctx, config) -> None:
    """For ttcore.LoadCachedOp: publish each parent input's
    accumulated golden into the session pool keyed by the input
    Tensor's globalId, so the sub-program's default pre-op finds it via
    the standard function-arg session-pool lookup."""
    if not ctx.checks_config.accumulation:
        return

    op = ctx.op
    pool = ctx.golden_tensor_pool
    asm_state = ctx.asm_state
    for mlir_input, rt_tensor_ref in zip(
        get_op_inputs(op), ctx.input_refs, strict=True
    ):
        check_shape_dtype(op, "mlir_vs_tensor_ref", mlir_input, rt_tensor_ref)
        parent_ssa = mlir_input.get_name(asm_state)
        golden = pool.get(parent_ssa)
        if golden is None:
            # Input is a function arg with no prior golden.
            continue
        publish_to_session_pool(ctx, rt_tensor_ref, golden)


@chisel_safe
def _subprogram_post_op(ctx, config) -> None:
    """For ttcore.LoadCachedOp: install each output's accumulated
    golden (published by the sub-program's post-op, by globalId) into the
    parent's golden pool. PCC was already done inside the sub-program."""
    if not ctx.checks_config.accumulation:
        return

    op = ctx.op
    asm_state = ctx.asm_state
    pool = ctx.golden_tensor_pool

    for mlir_output, output_ref in zip(
        get_op_outputs(op), ctx.output_refs, strict=True
    ):
        ssa = mlir_output.get_name(asm_state)
        promote_golden(ctx, op, ssa, output_ref)
