# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .ops import get_op_inputs
from .report import ChiselRecord, GoldenEvictedPayload
from .safety import chisel_safe


@chisel_safe
def _deallocate_pre_op(ctx, config) -> None:
    """Evict each input SSA from the golden + device pools; record one per evict."""
    golden_pool = ctx.golden_tensor_pool
    device_pool = ctx.device_tensor_pool
    asm_state = ctx.asm_state
    op = ctx.op
    for inp in get_op_inputs(op):
        ssa = inp.get_name(asm_state)
        # Drop from the device cache regardless of whether the golden pool
        # had an entry - the host copy is stale once the tensor is freed.
        device_pool.pop(ssa, None)
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
