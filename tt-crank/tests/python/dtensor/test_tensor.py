# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DTensor tensor distribution & manipulation on the tt backend.
"""

import gc

import pytest
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from tt_crank.torch import _native

pytestmark = pytest.mark.multichip


def test_distribute_replicate_round_trip(tt_pg) -> None:
    """All-Replicate placement: `distribute_tensor` round-trips through
    `full_tensor()` unchanged.
    """
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))

    t = torch.randn(32, 64, dtype=torch.bfloat16)

    # distribute → broadcast under the hood
    dt = distribute_tensor(t.to("tt"), mesh, [Replicate()])
    assert tuple(dt._local_tensor.shape) == (
        32,
        64,
    ), f"replicate local should match global, got {dt._local_tensor.shape}"

    full = dt.full_tensor().cpu()
    torch.testing.assert_close(full, t, atol=0.05, rtol=0.05)


def test_reshape_sharded_stays_sharded(tt_pg) -> None:
    """Reshape compatible with the sharding keeps Shard(0) and runs our view kernel
    per-shard.
    """
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))

    # Distinct data per shard: row block i (chip i) is all (i+1), bf16-exact.
    rows, cols = 32 * n, 64
    x = torch.empty(rows, cols, dtype=torch.bfloat16)
    for i in range(n):
        x[i * 32 : (i + 1) * 32] = float(i + 1)

    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])
    assert tuple(dx._local_tensor.shape) == (32, cols)

    # Reshape that changes the sharded dim 0 ([rows, cols] -> [rows//2, cols*2]).
    # Each chip's contiguous slab maps to a contiguous output block, so DTensor
    # keeps Shard(0) and runs the local view per-shard — no redistribution.
    dy = dx.reshape(rows // 2, cols * 2)
    assert dy._spec.placements == (
        Shard(0),
    ), f"expected Shard(0), got {dy._spec.placements}"
    assert tuple(dy._local_tensor.shape) == (
        16,
        cols * 2,
    ), f"local view should be per-shard, got {tuple(dy._local_tensor.shape)}"

    # full_tensor() all-gathers the shards back; a shard-0 collapse would make
    # every block equal chip 0's data (all 1.0).
    full = dy.full_tensor().cpu()
    assert torch.equal(
        full, x.reshape(rows // 2, cols * 2)
    ), "reconstructed reshape must match the global reshape"


def test_reshape_incompatible_with_sharding_raises(tt_pg) -> None:
    """A reshape incompatible with the sharding is DTensor's job to guard, not ours:
    under strict view it raises before our view kernel runs. We pin that boundary —
    if a future torch (pytorch #178616) switches `reshape` to auto-redistribute
    instead, this fails loudly.
    """
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("tp",))

    # Shard(1): chip c owns columns [32c:32c+32]. Reshaping to [rows*n, 32]
    # interleaves those columns in row-major order -> incompatible with Shard(1).
    x = torch.randn(32, 32 * n, dtype=torch.bfloat16)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(1)])

    with pytest.raises(RuntimeError, match="redistribution"):
        dx.reshape(32 * n, 32)


def _rss_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096


def test_empty_on_mesh_allocates_no_host_memory(tt_pg) -> None:
    """`empty` on a multi-chip mesh used to allocate one owned host copy per
    chip (mesh_size x the tensor) before any data existed. It is deferred now:
    no host memory until first read, and the read itself is one shared shard,
    not one copy per chip.
    """
    n = torch.tt.num_chips()
    torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))

    shape = (4096, 8192)  # 64 MiB in bf16
    nbytes = 4096 * 8192 * 2
    gc.collect()
    before = _rss_bytes()
    t = torch.empty(shape, device="tt", dtype=torch.bfloat16)
    assert _rss_bytes() - before < nbytes // 2
    assert not _native.tensor_storage_materialized(t)

    # First read materializes zeros once for the whole mesh: one shared host
    # shard plus the CPU result tensor (~2x), where the per-chip copies gave
    # (n + 1)x.
    host = t.cpu()
    grown = _rss_bytes() - before
    assert (
        grown < 2.5 * nbytes
    ), f"materializing grew RSS by {grown / 2**20:.0f} MiB for {n} chips"
    assert torch.equal(host, torch.zeros(shape, dtype=torch.bfloat16))


def test_from_local_on_empty_then_write(tt_pg) -> None:
    """`DTensor.from_local` over a fresh `empty` local (the placement pattern
    `distribute_module` / optimizer state init use): a later write lands and
    `full_tensor` sees every chip's shard. (from_local's replicate check reads
    the local tensor, so the placeholder is allowed to materialize here.)
    """
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))

    local = torch.empty((32, 64), device="tt", dtype=torch.bfloat16)
    dt = DTensor.from_local(local, mesh, [Replicate()])

    src = torch.randn((32, 64), dtype=torch.bfloat16)
    dt._local_tensor.copy_(src)
    torch.testing.assert_close(dt.full_tensor().cpu(), src, atol=0, rtol=0)

    # Sharded from_local on an untouched empty reads zeros on every chip.
    sharded = DTensor.from_local(
        torch.empty((32, 64), device="tt", dtype=torch.bfloat16), mesh, [Shard(0)]
    )
    full = sharded.full_tensor().cpu()
    assert tuple(full.shape) == (32 * n, 64)
    assert torch.equal(full, torch.zeros_like(full))
