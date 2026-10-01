# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Ops that take the CPU fallback must keep a sharded multi-device tensor sharded.

A tt tensor spans every chip of the mesh and its `.cpu()` is chip 0's slab only.
The fallback used to compute on that slab and upload the result replicated, so
any fallback op on a `Shard(0)` DTensor silently turned into "chip 0's shard on
every chip". These tests pin the per-shard behaviour with per-chip-distinct data,
where a shard-0 collapse makes every block equal chip 0's.
"""

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

pytestmark = pytest.mark.multichip


def _blocks(n: int, rows: int = 32, cols: int = 64) -> torch.Tensor:
    """bf16 tensor whose row block i (chip i under Shard(0)) is all (i + 1)."""
    x = torch.empty(rows * n, cols, dtype=torch.bfloat16)
    for i in range(n):
        x[i * rows : (i + 1) * rows] = float(i + 1)
    return x


def _mesh():
    n = torch.tt.num_chips()
    return n, torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))


def test_dtype_cast_keeps_shards(tt_pg) -> None:
    """`_to_copy` (`.float()`) is not a native tt kernel; it must cast every chip's slab."""
    n, mesh = _mesh()
    x = _blocks(n)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])

    dy = dx.float()
    assert dy._spec.placements == (Shard(0),)
    assert torch.equal(dy.full_tensor().cpu(), x.float())


def test_copy_with_dtype_mismatch_keeps_shards(tt_pg) -> None:
    """`dst.copy_(src)` across dtypes goes through `_copy_from`'s CPU conversion, per shard."""
    n, mesh = _mesh()
    x = _blocks(n)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])
    dst = distribute_tensor(
        torch.zeros_like(x, dtype=torch.float32).to("tt"), mesh, [Shard(0)]
    )

    dst.copy_(dx)
    assert torch.equal(dst.full_tensor().cpu(), x.float())


def test_pointwise_fallback_keeps_shards(tt_pg) -> None:
    """A plain out-of-place fallback op (erf has no tt kernel) runs per shard."""
    n, mesh = _mesh()
    x = _blocks(n)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])

    dy = torch.erf(dx / 4)
    assert dy._spec.placements == (Shard(0),)
    torch.testing.assert_close(
        dy.full_tensor().cpu(), torch.erf(x / 4), atol=0.05, rtol=0.05
    )


def test_log_softmax_fallback_keeps_shards(tt_pg) -> None:
    """`_log_softmax` (the loss path of every LM) falls back; per-row-block distinct data."""
    n, mesh = _mesh()
    rows, cols = 32, 64
    x = torch.randn(rows * n, cols, dtype=torch.bfloat16)
    x[:, 0] += (
        torch.arange(rows * n, dtype=torch.bfloat16) / 8
    )  # distinct rows, no ties
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])

    dy = torch.log_softmax(dx, dim=-1)
    assert dy._spec.placements == (Shard(0),)
    torch.testing.assert_close(
        dy.full_tensor().cpu(), torch.log_softmax(x, dim=-1), atol=0.05, rtol=0.05
    )


def test_inplace_fallback_writes_every_shard(tt_pg) -> None:
    """An in-place fallback op (`tanh_` has no tt kernel) must mutate every chip's slab."""
    n, mesh = _mesh()
    x = _blocks(n) / 4
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])

    dx.tanh_()
    torch.testing.assert_close(dx.full_tensor().cpu(), x.tanh(), atol=0.05, rtol=0.05)


def test_replicated_fallback_still_replicated(tt_pg) -> None:
    """Replicated operands take the single-run path and stay replicated."""
    n, mesh = _mesh()
    x = torch.randn(32, 64, dtype=torch.bfloat16)
    dx = distribute_tensor(x.to("tt"), mesh, [Replicate()])

    dy = torch.erf(dx.float())
    assert dy._spec.placements == (Replicate(),)
    torch.testing.assert_close(
        dy.full_tensor().cpu(), torch.erf(x.float()), atol=0.05, rtol=0.05
    )


def test_mixed_sharded_and_replicated_operands(tt_pg) -> None:
    """Binary fallback with one sharded and one replicated operand: the replicated
    slab pairs with every shard."""
    n, mesh = _mesh()
    x = _blocks(n)
    w = torch.randn(64, dtype=torch.bfloat16)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])
    dw = distribute_tensor(w.to("tt"), mesh, [Replicate()])

    # atan2 has no tt kernel and a DTensor pointwise strategy.
    dy = torch.atan2(dx, dw)
    assert dy._spec.placements == (Shard(0),)
    torch.testing.assert_close(
        dy.full_tensor().cpu(), torch.atan2(x, w), atol=0.05, rtol=0.05
    )
