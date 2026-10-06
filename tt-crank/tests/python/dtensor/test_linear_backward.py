# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DTensor autograd through the tt `linear_backward` and `matmul_backward` ops.

tt keeps both backward ops as leaf ops, so DTensor needs the sharding strategies in
`_sharding.py` to propagate through them at all; without one the propagator raises. The tests
run a whole forward + loss + backward with the inputs placed the way DP / Megatron-TP place them
and check the gradients against CPU autograd. The gradient placements are asserted too: a weight
gradient must come back in the weight's own placement (`Partial` over a data-parallel dim), never
`Replicate` -- that would mean the weight was all-gathered to compute it.
"""

import copy
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import (
    Partial,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)

pytestmark = pytest.mark.multichip

# parallel mode -> (mesh dim names, x placement, fc1 column-parallel / fc2 row-parallel param placements)
_PARALLEL = {
    "dp": (("dp",), [Shard(0)], {}),
    "tp": (
        ("tp",),
        [Replicate()],
        {
            "fc1.weight": [Shard(0)],
            "fc1.bias": [Shard(0)],
            "fc2.weight": [Shard(1)],
            "fc2.bias": [Replicate()],
        },
    ),
    "dp_tp": (
        ("dp", "tp"),
        [Shard(0), Replicate()],
        {
            "fc1.weight": [Replicate(), Shard(0)],
            "fc1.bias": [Replicate(), Shard(0)],
            "fc2.weight": [Replicate(), Shard(1)],
            "fc2.bias": [Replicate(), Replicate()],
        },
    ),
}


def _assert_grad(got, ref: torch.Tensor, placements) -> None:
    assert got is not None, "no gradient"
    assert got.placements == tuple(placements), got.placements
    # bf16 device result vs fp32 CPU autograd; a Partial gradient is rounded to bf16 per chip and
    # summed across chips in bf16, so the tolerance follows the gradient's own scale.
    ref = ref.float()
    torch.testing.assert_close(
        got.full_tensor().cpu().float(), ref, atol=0.05 * ref.std().item(), rtol=0.1
    )


@pytest.mark.parametrize("mode", ["eager", "compile"])
@pytest.mark.parametrize("parallel", list(_PARALLEL), ids=list(_PARALLEL))
def test_linear_backward(tt_pg, mesh_2d_shape, parallel: str, mode: str) -> None:
    """Two-layer MLP forward + MSE loss + backward with the parameters placed per `parallel`:
    DP (batch sharded, params replicated), Megatron TP (fc1 column-parallel, fc2 row-parallel),
    or both on a 2-D mesh. Every param gradient must match CPU and keep the param's placement,
    with `Replicate` turned into `Partial` on the dp dim. No activation between the layers: a
    bf16-vs-fp32 relu flip near zero would swamp the gradient check without testing anything."""
    names, x_placement, param_placements = _PARALLEL[parallel]
    shape = mesh_2d_shape if len(names) == 2 else (torch.tt.num_chips(),)
    mesh = torch.tt.init_device_mesh(shape, mesh_dim_names=names)
    size = dict(zip(names, shape))
    batch, feat, hidden, classes = (
        32 * size.get("dp", 1),
        32,
        32 * size.get("tp", 1),
        32,
    )

    model = nn.Sequential(
        OrderedDict(fc1=nn.Linear(feat, hidden), fc2=nn.Linear(hidden, classes))
    ).to(torch.bfloat16)
    x = torch.randn(batch, feat, dtype=torch.bfloat16)
    target = torch.randn(batch, classes, dtype=torch.bfloat16)

    ref = copy.deepcopy(model).float()
    F.mse_loss(ref(x.float()), target.float()).backward()

    def partition_fn(name: str, module: nn.Module, device_mesh) -> None:
        for pname, p in module.named_parameters(recurse=False):
            placements = param_placements.get(
                f"{name}.{pname}", [Replicate()] * mesh.ndim
            )
            module.register_parameter(
                pname, nn.Parameter(distribute_tensor(p, device_mesh, placements))
            )

    dmodel = distribute_module(model.to("tt"), mesh, partition_fn=partition_fn)
    dx = distribute_tensor(x.to("tt"), mesh, x_placement)
    fwd = dmodel
    if mode == "compile":
        fwd = torch.compile(dmodel, backend="tt")
    out = fwd(dx)
    F.mse_loss(out.full_tensor(), target.to("tt")).backward()

    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    for name, p in dmodel.named_parameters():
        expected = [
            Partial() if dim == "dp" and pl.is_replicate() else pl
            for dim, pl in zip(names, p.placements)
        ]
        _assert_grad(p.grad, ref_grads[name], expected)


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_matmul_backward_head_parallel(tt_pg, mode: str) -> None:
    """Attention scores `q @ k^T` with the head dim (a batch dim of the matmul) sharded: both
    gradients keep the head shard, no collective. Batch 1 because the forward is torch's `matmul`
    decomposition (bmm over a merged batch*heads dim), which can only carry the head shard through
    its reshapes when there is nothing to merge it with."""
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("tp",))
    batch, heads, seq, dim = 1, 2 * n, 32, 64
    q = torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq, dim, dtype=torch.bfloat16)
    target = torch.randn(batch, heads, seq, seq, dtype=torch.bfloat16)

    def scores(a, b):
        return torch.matmul(a, b.transpose(-1, -2))

    refs = [t.float().requires_grad_(True) for t in (q, k)]
    F.mse_loss(scores(*refs), target.float()).backward()

    dq, dk = (
        distribute_tensor(t.to("tt"), mesh, [Shard(1)]).requires_grad_(True)
        for t in (q, k)
    )
    if mode == "compile":
        scores = torch.compile(scores, backend="tt")
    out = scores(dq, dk)
    assert out.placements == (Shard(1),), out.placements
    F.mse_loss(out.full_tensor(), target.to("tt")).backward()

    for got, ref in zip((dq, dk), refs):
        _assert_grad(got.grad, ref.grad, [Shard(1)])
