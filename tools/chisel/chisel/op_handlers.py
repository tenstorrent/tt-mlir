# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from _ttmlir_runtime import runtime as tt_runtime

from golden import GoldenMapTensor

from .ops import get_op_inputs, get_op_outputs, is_cpu_hoist_call
from .report import (
    ChiselRecord,
    GoldenEvictedPayload,
    NoGoldenPayload,
    NumericsMode,
)
from .safety import chisel_safe
from .utils import (
    golden_to_runtime_tensor,
    invalidate_device_cache,
    promote_golden,
    publish_to_session_pool,
    tensor_to_golden,
)
from .validators import check_shape_dtype, emit_pcc, validate_and_retrieve_tensor


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


def _invoke_cpu_op(ctx, inputs: List[GoldenMapTensor]) -> List[GoldenMapTensor]:
    """Re-invoke the dylib behind the current CpuOp with `inputs` as goldens.

    Single-device only: each output is wrapped back as a GoldenMapTensor over
    ctx.mesh_shape, which the assert pins to a 1x1 mesh.
    """
    assert ctx.mesh_shape == (1, 1), (
        "cpu-hoist dylib re-invocation is single-device only; "
        f"got mesh_shape {ctx.mesh_shape}"
    )
    rt_inputs = [golden_to_runtime_tensor(golden) for golden in inputs]
    rt_outputs = tt_runtime.invoke_cpu_op(
        ctx.rt_program_context, ctx.rt_op_context, rt_inputs
    )
    return [tensor_to_golden(rt, ctx.mesh_shape) for rt in rt_outputs]


@chisel_safe
def _cpu_hoist_pre_op(ctx, config) -> None:
    """func.CallOp PRE.

    For CPU-hoisted calls, promote any input SSA not already in the golden pool
    by seeding it from the live device tensor (typically a function arg). For
    non-hoisted func.CallOps, preserve the previous no_golden behavior.
    """
    op = ctx.op
    if not is_cpu_hoist_call(op):
        # TODO(ndrakulic): handle plain (non-cpu-hoisted) func.CallOps the same
        # way as _subprogram_pre_op/_subprogram_post_op handle ttcore.LoadCachedOp
        # - chain the parent's accumulated goldens through the callee via the
        # cross-program globalId session pool (publish inputs by globalId in PRE,
        # read callee outputs by globalId in POST) instead of dropping to
        # no_golden here.
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="golden_not_implemented",
                payload=NoGoldenPayload(),
            )
        )
        return

    asm_state = ctx.asm_state
    pool = ctx.golden_tensor_pool
    for mlir_input, rt_ref in zip(get_op_inputs(op), ctx.input_refs, strict=True):
        ssa = mlir_input.get_name(asm_state)
        if ssa in pool:
            continue
        promote_golden(ctx, op, ssa, rt_ref)


@chisel_safe
def _cpu_hoist_post_op(ctx, config) -> None:
    """func.CallOp POST.

    For CPU-hoisted calls, invoke the dylib with accumulated goldens for the
    input SSAs, seed each output SSA into the golden pool, and PCC-check each
    output against the live device tensor. For non-hoisted calls this is a
    no-op.
    """
    op = ctx.op
    if not is_cpu_hoist_call(op):
        return
    if not ctx.checks_config.accumulation:
        return

    asm_state = ctx.asm_state
    pool = ctx.golden_tensor_pool
    output_vals = get_op_outputs(op)
    output_ssas = [v.get_name(asm_state) for v in output_vals]

    input_goldens = [pool[v.get_name(asm_state)] for v in get_op_inputs(op)]
    accum_outs = _invoke_cpu_op(ctx, input_goldens)

    for ssa, golden in zip(output_ssas, accum_outs, strict=True):
        pool[ssa] = golden

    for mlir_out, out_ref, ssa, accum_out in zip(
        output_vals, ctx.output_refs, output_ssas, accum_outs, strict=True
    ):
        device_tensor = validate_and_retrieve_tensor(ctx, mlir_out, out_ref)
        emit_pcc(
            ctx,
            op,
            ssa,
            mlir_out,
            accum_out,
            device_tensor,
            mode=NumericsMode.ACCUMULATED,
            skip_pcc=config.skip_pcc,
        )
