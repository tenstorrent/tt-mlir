"""DTensor Linear / matmul on the tt backend: data-parallel and row-parallel (TP).

- DP: replicated weight/bias + Shard(0) input → broadcast + scatter; F.linear
  runs per-chip (Shard(0) output), and `full_tensor()` all-gathers the result.
- Row-parallel (Megatron): X and W sharded along the contraction dim K, so each
  chip computes a partial sum (Shard(K) × Shard(K) → Partial); `full_tensor()`
  redistributes Partial → Replicate, firing `allreduce` → `ttir.all_reduce`.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

from tt_kurbla.torch import _native

pytestmark = pytest.mark.multichip


def test_linear_replicated_weight_sharded_input(tt_pg) -> None:
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))

    batch = 32 * n
    in_features, out_features = 32, 64
    x = torch.randn(batch, in_features, dtype=torch.bfloat16)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    bias = torch.randn(out_features, dtype=torch.bfloat16)
    expected = F.linear(x, weight, bias)  # [batch, out_features]

    # Replicated weight + bias → broadcast; Shard(0) input → scatter.
    dweight = distribute_tensor(weight.to("tt"), mesh, [Replicate()])
    dbias = distribute_tensor(bias.to("tt"), mesh, [Replicate()])
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])

    desc_w = _native.describe_tensor(dweight._local_tensor)
    assert "PlacementReplicate()" in desc_w, f"weight should be replicated:\n{desc_w}"

    desc_x = _native.describe_tensor(dx._local_tensor)
    assert "PlacementShard(0)" in desc_x, f"input should be Shard(0):\n{desc_x}"

    # Shard(0) input × Replicate weight → Shard(0) output (batch-dim sharding).
    dout = F.linear(dx, dweight, dbias)
    assert dout._spec.placements == (Shard(0),), \
        f"expected output placement (Shard(0),), got {dout._spec.placements}"
    assert tuple(dout._local_tensor.shape) == (batch // n, out_features), \
        f"per-rank output shape mismatch: {dout._local_tensor.shape}"

    desc_out = _native.describe_tensor(dout._local_tensor)
    assert "PlacementShard(0)" in desc_out, f"output topology should be Shard(0):\n{desc_out}"

    # full_tensor() redistributes Shard(0) → Replicate via allgather.
    full = dout.full_tensor()
    assert full.device.type == "tt", f"full_tensor should stay on tt, got {full.device}"
    assert tuple(full.shape) == (batch, out_features), \
        f"full_tensor shape mismatch: got {tuple(full.shape)}, expected ({batch}, {out_features})"

    desc_full = _native.describe_tensor(full)
    assert "PlacementReplicate()" in desc_full, f"full_tensor topology should be Replicate:\n{desc_full}"
    assert "PlacementShard" not in desc_full, f"unexpected shard placement:\n{desc_full}"

    full_cpu = full.cpu()
    torch.testing.assert_close(full_cpu, expected, atol=0.1, rtol=0.1)


def test_linear_row_parallel(tt_pg) -> None:
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("tp",))

    # Per-chip K must be >= 32 for ttnn tile alignment, and the output dim
    # must be tile-count-divisible by `n` so tt-mlir's all_reduce →
    # reduce_scatter+all_gather workaround takes its reduce_scatter path.
    #
    # W is laid out (K, N) — no transpose needed. Our backend has no
    # placement-aware aten::t, so a transpose fallback would collapse to
    # shard 0 and break the row-parallel math.
    out_features = 64
    batch = 32 * n
    in_features = 32 * n
    x = torch.randn(batch, in_features, dtype=torch.bfloat16)
    w_kn = torch.randn(in_features, out_features, dtype=torch.bfloat16)
    expected = x @ w_kn

    # Shard along the contraction dim on both sides:
    #   x    [B, K]  → Shard(1)
    #   w_kn [K, N]  → Shard(0)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(1)])
    dweight = distribute_tensor(w_kn.to("tt"), mesh, [Shard(0)])

    # Underlying topology should be Shard (any dim — our scatter hardcodes
    # the tensor shard dim to 0 in the topology metadata).
    assert "PlacementShard" in _native.describe_tensor(dx._local_tensor)
    assert "PlacementShard" in _native.describe_tensor(dweight._local_tensor)

    # Shard(K)×Shard(K) → Partial output (per-chip partial sums).
    dout = dx @ dweight
    assert dout._spec.placements == (Partial(),), \
        f"expected Partial output, got {dout._spec.placements}"

    # full_tensor() redistributes Partial → Replicate, firing all_reduce.
    full = dout.full_tensor()
    assert full.device.type == "tt"
    assert tuple(full.shape) == (batch, out_features)
    assert "PlacementReplicate()" in _native.describe_tensor(full)

    torch.testing.assert_close(full.cpu(), expected, atol=0.2, rtol=0.1)
