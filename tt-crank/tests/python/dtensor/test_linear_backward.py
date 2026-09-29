# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DTensor backward through the tt `linear` and `matmul` leaf ops.

tt keeps `linear_backward` and `matmul_backward` as ops, so DTensor needs the sharding
strategies in `_sharding.py` to propagate through them at all; without one the propagator
raises. Each test drives one strategy family with inputs already placed the way that family
wants them, checks the gradient placements it promises (no collective on the weight gradient
for TP, `Partial` sums for DP) and the values against CPU autograd.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.tensor import Partial, Replicate, Shard, distribute_tensor

pytestmark = pytest.mark.multichip

_DT = torch.bfloat16


def _mesh():
    n = torch.tt.num_chips()
    return n, torch.tt.init_device_mesh((n,), mesh_dim_names=("d",))


def _run(mode: str, fn, *args):
    if mode == "eager":
        return fn(*args)
    torch._dynamo.reset()
    return torch.compile(fn, backend="tt", fullgraph=True)(*args)


def _check_grad(name: str, got, ref: torch.Tensor, placement) -> None:
    assert got is not None, f"no gradient for {name}"
    assert got.placements == (
        placement,
    ), f"{name}.grad placed {got.placements}, expected ({placement},)"
    torch.testing.assert_close(got.full_tensor().cpu().float(), ref, atol=0.1, rtol=0.1)


# (x, weight, bias, grad_output) placements going in ->
# (grad_x, grad_weight, grad_bias) placements the family promises
_LINEAR_FAMILIES = {
    "dp": (
        (Shard(0), Replicate(), Replicate(), Shard(0)),
        (Shard(0), Partial(), Partial()),
    ),
    "column_tp": (
        (Replicate(), Shard(0), Shard(0), Shard(1)),
        (Partial(), Shard(0), Shard(0)),
    ),
    # bias is [out] and row TP shards in_features, so the bias stays replicated
    "row_tp": (
        (Shard(1), Shard(1), Replicate(), Replicate()),
        (Shard(1), Shard(1), Replicate()),
    ),
}


@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.parametrize("family", list(_LINEAR_FAMILIES), ids=list(_LINEAR_FAMILIES))
def test_linear_backward_sharding(tt_pg, family: str, mode: str) -> None:
    """DP shards the batch, column TP the out_features, row TP the in_features. In every family the
    weight gradient comes out with the weight's own placement (or `Partial` for DP), never
    `Replicate` -- a `Replicate` would mean the weight was all-gathered to compute it."""
    n, mesh = _mesh()
    (x_pl, w_pl, b_pl, g_pl), (gx_pl, gw_pl, gb_pl) = _LINEAR_FAMILIES[family]
    batch, in_features, out_features = 8 * n, 32 * n, 32 * n
    x = torch.randn(batch, in_features, dtype=_DT)
    w = torch.randn(out_features, in_features, dtype=_DT)
    b = torch.randn(out_features, dtype=_DT)
    grad_out = torch.randn(batch, out_features, dtype=_DT)

    refs = [t.float().clone().requires_grad_(True) for t in (x, w, b)]
    F.linear(*refs).backward(grad_out.float())

    dx = distribute_tensor(x.to("tt"), mesh, [x_pl]).requires_grad_(True)
    dw = distribute_tensor(w.to("tt"), mesh, [w_pl]).requires_grad_(True)
    db = distribute_tensor(b.to("tt"), mesh, [b_pl]).requires_grad_(True)
    dg = distribute_tensor(grad_out.to("tt"), mesh, [g_pl])

    out = _run(mode, F.linear, dx, dw, db)
    out.backward(dg)

    for name, got, ref, pl in zip("xwb", (dx, dw, db), refs, (gx_pl, gw_pl, gb_pl)):
        _check_grad(name, got.grad, ref.grad, pl)


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_matmul_backward_batch_parallel(tt_pg, mode: str) -> None:
    """Attention-shaped batched matmul sharded on the head dim (a batch dim of the matmul): both
    gradients keep the head shard, no collective. Batch 1 because the forward is torch's `matmul`
    decomposition (bmm over a merged batch*heads dim), which can only carry the head shard through
    its reshapes when there is nothing to merge it with."""
    n, mesh = _mesh()
    batch, heads, seq, dim = 1, 2 * n, 32, 64
    q = torch.randn(batch, heads, seq, dim, dtype=_DT)
    k = torch.randn(batch, heads, seq, dim, dtype=_DT)
    grad_out = torch.randn(batch, heads, seq, seq, dtype=_DT)

    refs = [t.float().clone().requires_grad_(True) for t in (q, k)]
    torch.matmul(refs[0], refs[1].transpose(-1, -2)).backward(grad_out.float())

    dq, dk = (
        distribute_tensor(t.to("tt"), mesh, [Shard(1)]).requires_grad_(True)
        for t in (q, k)
    )
    dg = distribute_tensor(grad_out.to("tt"), mesh, [Shard(1)])

    out = _run(mode, lambda a, b: torch.matmul(a, b.transpose(-1, -2)), dq, dk)
    assert out.placements == (Shard(1),), out.placements
    out.backward(dg)

    for name, got, ref in zip("qk", (dq, dk), refs):
        _check_grad(name, got.grad, ref.grad, Shard(1))
