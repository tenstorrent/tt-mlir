import gc

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import (
    Partial,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)

from _models import MNISTLinear

pytestmark = pytest.mark.multichip

# Each parallel mode is exercised both ways: `fused=False` runs the
# redistribution as a separate eager TTIR module after the compiled forward;
# `fused=True` captures it inside the compiled graph as an in-graph functional
# collective lowered to a `ttir.*` CCL (one fused module, no eager PG dispatch).
FUSED = pytest.mark.parametrize("fused", [False, True], ids=["eager_ccl", "fused"])


def _shard_param(module: nn.Module, name: str, mesh, placements) -> None:
    """Replace `module.<name>` with its DTensor distributed per `placements`.
    Used inside a `distribute_module` `partition_fn`."""
    p = getattr(module, name)
    module.register_parameter(name, nn.Parameter(distribute_tensor(p, mesh, placements), requires_grad=False))


def _compiled(model: nn.Module, dx, mesh, *, fused: bool) -> torch.Tensor:
    """Run `model` through the tt compile backend on a DTensor input and bring
    the global result back to CPU.

    `fused=False`: the redistribution (`full_tensor()`) happens *outside* the
    compiled region, so its collective dispatches eagerly through the process
    group as a separate TTIR module.

    `fused=True`: the redistribution to `Replicate` happens *inside* the
    compiled function, so DTensor emits the collective as an in-graph
    `_c10d_functional.*` op that the backend lowers to a `ttir.*` CCL — the
    whole forward + collective fuse into one TTIR module, no eager PG dispatch.
    """
    with torch.no_grad():
        if fused:

            def fn(x):
                return model(x).redistribute(mesh, [Replicate()] * mesh.ndim)

            out = torch.compile(fn, backend="tt")(dx)
            # Already Replicate → to_local() is the full result, no collective.
            return out.to_local().cpu()

        out = torch.compile(model, backend="tt")(dx)
        return out.full_tensor().cpu()


@FUSED
def test_mnist_dp_compile(tt_pg, fused: bool) -> None:
    """Data-parallel MNISTLinear: `distribute_module` replicates every
    parameter, the input batch is sharded. The whole two-layer forward lowers
    to one TTIR module per chip; the output all-gather (Shard(0)→Replicate)
    resolves the global result."""
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))

    batch, feat, hidden, classes = 32 * n, 32 * 32, 128, 32
    model = MNISTLinear(feat, hidden, classes).to(torch.bfloat16)
    x = torch.randn(batch, feat, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = model(x)

    # No partition_fn → distribute_module replicates all params across the mesh.
    dmodel = distribute_module(model.to("tt"), mesh)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0)])

    full = _compiled(dmodel, dx, mesh, fused=fused)

    assert tuple(full.shape) == (batch, classes)
    torch.testing.assert_close(full, expected, atol=0.05, rtol=0.1)


@FUSED
def test_mnist_tp_compile(tt_pg, fused: bool) -> None:
    """Tensor-parallel (Megatron MLP) MNISTLinear: `partition_fn` shards fc1
    column-parallel and fc2 row-parallel, input replicated. fc2's contraction
    over the sharded dim yields a Partial output; the all-reduce (Partial→
    Replicate) resolves the global result."""
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("tp",))

    # hidden is sharded across chips (fc1 out / fc2 in) → must divide by n and
    # stay tile-aligned per chip (>= 32).
    batch, feat, hidden, classes = 32, 32 * 32, 32 * n, 32
    model = MNISTLinear(feat, hidden, classes).to(torch.bfloat16)
    x = torch.randn(batch, feat, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = model(x)

    def partition_fn(name: str, module: nn.Module, device_mesh) -> None:
        if name == "fc1":  # column-parallel: shard the output (hidden) dim
            _shard_param(module, "weight", device_mesh, [Shard(0)])
            _shard_param(module, "bias", device_mesh, [Shard(0)])
        elif name == "fc2":  # row-parallel: shard the input (hidden) dim
            _shard_param(module, "weight", device_mesh, [Shard(1)])
            _shard_param(module, "bias", device_mesh, [Replicate()])

    dmodel = distribute_module(model.to("tt"), mesh, partition_fn=partition_fn)
    dx = distribute_tensor(x.to("tt"), mesh, [Replicate()])

    # Confirm fc2 really produces a Partial output before the redistribute.
    with torch.no_grad():
        eager_out = dmodel(dx)
    assert eager_out._spec.placements == (Partial(),), \
        f"expected Partial output from row-parallel fc2, got {eager_out._spec.placements}"

    full = _compiled(dmodel, dx, mesh, fused=fused)

    assert tuple(full.shape) == (batch, classes)
    torch.testing.assert_close(full, expected, atol=0.1, rtol=0.1)


# ===== 2-D parallel (DP × TP) MNISTLinear =====
#
# Data-parallel on mesh axis 0 (dp), Megatron tensor-parallel on axis 1 (tp).
# fc2's row-parallel contraction yields a `Partial` on tp while the batch stays
# `Shard(0)` on dp, so resolving to the global result fires *two* collectives,
# each scoped to its own axis: all-reduce over tp + all-gather over dp.


def _build_2d_dp_tp(mesh):
    """A `distribute_module` partition_fn placing MNISTLinear in 2-D DP×TP:
    weights replicated over dp, Megatron-sharded over tp."""

    def partition_fn(name: str, module: nn.Module, device_mesh) -> None:
        if name == "fc1":  # column-parallel on tp, replicated on dp
            _shard_param(module, "weight", device_mesh, [Replicate(), Shard(0)])
            _shard_param(module, "bias", device_mesh, [Replicate(), Shard(0)])
        elif name == "fc2":  # row-parallel on tp, replicated on dp
            _shard_param(module, "weight", device_mesh, [Replicate(), Shard(1)])
            _shard_param(module, "bias", device_mesh, [Replicate(), Replicate()])

    return partition_fn


@FUSED
def test_mnist_2d_dp_tp_compile(tt_pg, mesh_2d_shape, fused: bool) -> None:
    """2-D DP×TP MNISTLinear; resolving the global result fires two collectives
    (all-reduce over tp, all-gather over dp), each scoped to its own axis."""
    gc.collect()
    dp, tp = mesh_2d_shape
    mesh = torch.tt.init_device_mesh((dp, tp), mesh_dim_names=("dp", "tp"))

    # batch÷dp and hidden÷tp must stay tile-aligned (>= 32 per chip).
    batch, feat, hidden, classes = 32 * dp, 32 * 32, 32 * tp, 32
    model = MNISTLinear(feat, hidden, classes).to(torch.bfloat16)
    x = torch.randn(batch, feat, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = model(x)

    dmodel = distribute_module(model.to("tt"), mesh, partition_fn=_build_2d_dp_tp(mesh))
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0), Replicate()])  # DP batch shard

    full = _compiled(dmodel, dx, mesh, fused=fused)

    assert tuple(full.shape) == (batch, classes)
    torch.testing.assert_close(full, expected, atol=0.2, rtol=0.1)
