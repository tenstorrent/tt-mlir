# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .ops import get_op_inputs
from .report import ChiselRecord, GoldenEvictedPayload
from .safety import chisel_safe
from .utils import invalidate_device_cache


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
