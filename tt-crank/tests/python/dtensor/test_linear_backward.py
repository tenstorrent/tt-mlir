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
def _linear_family(family: str, ndim: int):
    last = ndim - 1  # x and grad_output's feature dim
    return {
        "dp": (
            (Shard(0), Replicate(), Replicate(), Shard(0)),
            (Shard(0), Partial(), Partial()),
        ),
        "column_tp": (
            (Replicate(), Shard(0), Shard(0), Shard(last)),
            (Partial(), Shard(0), Shard(0)),
        ),
        # bias is [out] and row TP shards in_features, so the bias stays replicated
        "row_tp": (
            (Shard(last), Shard(1), Replicate(), Replicate()),
            (Shard(last), Shard(1), Replicate()),
        ),
    }[family]


_LINEAR_FAMILIES = ("dp", "column_tp", "row_tp")


@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("ndim", [2, 3], ids=["x2d", "x3d"])
@pytest.mark.parametrize("family", _LINEAR_FAMILIES, ids=_LINEAR_FAMILIES)
def test_linear_backward_sharding(
    tt_pg, family: str, ndim: int, bias: bool, mode: str
) -> None:
    """DP shards the batch, column TP the out_features, row TP the in_features. In every family the
    weight gradient comes out with the weight's own placement (or `Partial` for DP), never
    `Replicate` -- a `Replicate` would mean the weight was all-gathered to compute it.

    `x3d` is the transformer shape `[batch, seq, in]`: the feature dim moves to 2 and the weight
    gradient reduces over two leading dims.

    `no_bias` runs with output_mask (True, True, False): torch's meta still returns a grad_bias
    tensor there, so the strategy must give it a spec or the compile path fails."""
    n, mesh = _mesh()
    (x_pl, w_pl, b_pl, g_pl), (gx_pl, gw_pl, gb_pl) = _linear_family(family, ndim)
    batch, in_features, out_features = 8 * n, 32 * n, 32 * n
    lead = (batch,) if ndim == 2 else (batch, 4)
    x = torch.randn(*lead, in_features, dtype=_DT)
    w = torch.randn(out_features, in_features, dtype=_DT)
    b = torch.randn(out_features, dtype=_DT)
    grad_out = torch.randn(*lead, out_features, dtype=_DT)

    refs = [t.float().clone().requires_grad_(True) for t in (x, w, b)]
    F.linear(*refs[: 2 + bias]).backward(grad_out.float())

    dx = distribute_tensor(x.to("tt"), mesh, [x_pl]).requires_grad_(True)
    dw = distribute_tensor(w.to("tt"), mesh, [w_pl]).requires_grad_(True)
    db = (
        distribute_tensor(b.to("tt"), mesh, [b_pl]).requires_grad_(True)
        if bias
        else None
    )
    dg = distribute_tensor(grad_out.to("tt"), mesh, [g_pl])

    out = _run(mode, F.linear, dx, dw, db)
    out.backward(dg)

    for name, got, ref, pl in zip("xwb", (dx, dw, db), refs, (gx_pl, gw_pl, gb_pl)):
        if got is not None:
            _check_grad(name, got.grad, ref.grad, pl)


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_linear_backward_2d_mesh(tt_pg, mesh_2d_shape, mode: str) -> None:
    """DP x column-TP on a 2-D mesh: the strategy is written per mesh dim and DTensor takes the
    product, so dim "dp" runs the DP family and dim "tp" the column-TP family at once.

                    x            weight       bias         grad_out
        in:    (S0, R)        (R, S0)      (R, S0)      (S0, S1)
        out:   (S0, P)        (P, S0)      (P, S0)                   = (grad_x, grad_w, grad_b)
    """
    dp, tp = mesh_2d_shape
    mesh = torch.tt.init_device_mesh((dp, tp), mesh_dim_names=("dp", "tp"))
    batch, in_features, out_features = 8 * dp, 32, 32 * tp
    x = torch.randn(batch, in_features, dtype=_DT)
    w = torch.randn(out_features, in_features, dtype=_DT)
    b = torch.randn(out_features, dtype=_DT)
    grad_out = torch.randn(batch, out_features, dtype=_DT)

    refs = [t.float().clone().requires_grad_(True) for t in (x, w, b)]
    F.linear(*refs).backward(grad_out.float())

    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0), Replicate()]).requires_grad_(
        True
    )
    dw = distribute_tensor(w.to("tt"), mesh, [Replicate(), Shard(0)]).requires_grad_(
        True
    )
    db = distribute_tensor(b.to("tt"), mesh, [Replicate(), Shard(0)]).requires_grad_(
        True
    )
    dg = distribute_tensor(grad_out.to("tt"), mesh, [Shard(0), Shard(1)])

    out = _run(mode, F.linear, dx, dw, db)
    out.backward(dg)

    expected = {
        "x": (Shard(0), Partial()),
        "w": (Partial(), Shard(0)),
        "b": (Partial(), Shard(0)),
    }
    for name, got, ref in zip("xwb", (dx, dw, db), refs):
        assert got.grad is not None, f"no gradient for {name}"
        assert (
            got.grad.placements == expected[name]
        ), f"{name}.grad placed {got.grad.placements}"
        torch.testing.assert_close(
            got.grad.full_tensor().cpu().float(), ref.grad, atol=0.1, rtol=0.1
        )


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


def test_matmul_backward_strategy_aligns_batch_dims_right() -> None:
    """Strategy-only (the tt matmul kernel rejects rank-mismatched batched inputs, so this cannot
    run on device). `q [1, H, S, D] @ kT [H, D, S]`: the head dim is 1 on q and grad but 0 on kT,
    and the size-1 batch dim of q is skipped rather than sharded.

        grad [1, H, S, S]   q [1, H, S, D]   kT [H, D, S]
        Shard(1)            Shard(1)         Shard(0)      ->  grad_q Shard(1), grad_k Shard(0)
    """
    from types import SimpleNamespace

    from tt_crank.torch._sharding import _matmul_backward_sharding

    spec = lambda *shape: SimpleNamespace(ndim=len(shape), shape=shape)
    grad, q, kT = spec(1, 4, 32, 32), spec(1, 4, 32, 64), spec(4, 64, 32)
    assert _matmul_backward_sharding(grad, q, kT, [True, True]) == [
        ([Replicate(), Replicate()], [Replicate(), Replicate(), Replicate(), None]),
        ([Shard(1), Shard(0)], [Shard(1), Shard(1), Shard(0), None]),
    ]


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_matmul_backward_dp_3d_by_2d(tt_pg, mode: str) -> None:
    """`x [B, S, D] @ w [D, E]` with the batch sharded: w has no batch dim, so it stays replicated
    and its gradient (summed over the batch) comes back `Partial`, no all-gather of x or grad.

        grad [B, S, E]   x [B, S, D]   w [D, E]
        Shard(0)         Shard(0)      Replicate   ->  grad_x Shard(0), grad_w Partial
    """
    n, mesh = _mesh()
    batch, seq, dim, out_dim = 2 * n, 32, 64, 32
    x = torch.randn(batch, seq, dim, dtype=_DT)
    w = torch.randn(dim, out_dim, dtype=_DT)
    grad_out = torch.randn(batch, seq, out_dim, dtype=_DT)

    refs = [t.float().clone().requires_grad_(True) for t in (x, w)]
    torch.matmul(*refs).backward(grad_out.float())

    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)]).requires_grad_(True)
    dw = distribute_tensor(w.to("tt"), mesh, [Replicate()]).requires_grad_(True)
    dg = distribute_tensor(grad_out.to("tt"), mesh, [Shard(0)])

    out = _run(mode, torch.matmul, dx, dw)
    assert out.placements == (Shard(0),), out.placements
    out.backward(dg)

    _check_grad("x", dx.grad, refs[0].grad, Shard(0))
    _check_grad("w", dw.grad, refs[1].grad, Partial())
