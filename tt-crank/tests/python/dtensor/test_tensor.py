"""DTensor tensor distribution & manipulation on the tt backend.
"""

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from tt_kurbla.torch import _native

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
    assert tuple(dt._local_tensor.shape) == (32, 64), \
        f"replicate local should match global, got {dt._local_tensor.shape}"

    # ttnn TensorTopology must be fully replicated over the N-device mesh.
    desc = _native.describe_tensor(dt._local_tensor)
    assert "PlacementReplicate()" in desc, f"expected replicated topology, got:\n{desc}"
    assert "PlacementShard" not in desc, f"unexpected sharded topology, got:\n{desc}"
    assert f"MeshShape([{n}])" in desc, f"expected distribution_shape [{n}], got:\n{desc}"
    coords_present = sum(f"MeshCoordinate([0, {i}])" in desc for i in range(n))
    assert coords_present == n, f"expected {n} mesh coords, got {coords_present} in:\n{desc}"

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
    assert dy._spec.placements == (Shard(0),), f"expected Shard(0), got {dy._spec.placements}"
    assert tuple(dy._local_tensor.shape) == (16, cols * 2), \
        f"local view should be per-shard, got {tuple(dy._local_tensor.shape)}"
    assert "PlacementShard(0)" in _native.describe_tensor(dy._local_tensor), \
        "reshape must preserve Shard topology, not collapse to Replicate"

    # full_tensor() all-gathers the shards back; a shard-0 collapse would make
    # every block equal chip 0's data (all 1.0).
    full = dy.full_tensor().cpu()
    assert torch.equal(full, x.reshape(rows // 2, cols * 2)), "reconstructed reshape must match the global reshape"


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


def test_copy_cpu_into_sharded_raises(tt_pg) -> None:
    """copy_(cpu→tt) rebuilds replicated storage — only rank 0's local data
    exists, so copying into a sharded local tensor would silently overwrite
    every shard with it. Must fail loudly; a replicated destination stays fine.
    """
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))

    x = torch.randn(32 * n, 64, dtype=torch.bfloat16)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])
    with pytest.raises(RuntimeError, match="destination is sharded"):
        dx._local_tensor.copy_(torch.zeros(32, 64, dtype=torch.bfloat16))

    # Replicated destination: in-place update is well-defined and must work.
    dw = distribute_tensor(x.to("tt"), mesh, [Replicate()])
    new_w = torch.randn(32 * n, 64, dtype=torch.bfloat16)
    dw._local_tensor.copy_(new_w)
    torch.testing.assert_close(dw.full_tensor().cpu(), new_w, atol=0.05, rtol=0.05)
