import pytest
import torch
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor

pytestmark = pytest.mark.multichip


@pytest.mark.parametrize("axis", [0, 1])
def test_all_gather_per_axis(tt_pg, mesh_2d_shape, axis: int) -> None:
    """Shard along one mesh axis, replicate the other; `full_tensor()` fires a
    single all-gather over that axis."""
    mesh = torch.tt.init_device_mesh(mesh_2d_shape, mesh_dim_names=("rows", "cols"))
    axis_size = mesh.shape[axis]

    x = torch.randn(32 * axis_size, 32, dtype=torch.bfloat16)
    placements = [Shard(0), Replicate()] if axis == 0 else [Replicate(), Shard(0)]
    dx = distribute_tensor(x.to("tt"), mesh, placements)

    full = dx.full_tensor().cpu()
    torch.testing.assert_close(full, x, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("axis", [0, 1])
def test_all_reduce_per_axis(tt_pg, mesh_2d_shape, axis: int) -> None:
    """Partial along one mesh axis; `full_tensor()` fires an all-reduce over
    it. Every chip holds the same local slab, so the result is local *
    axis_size."""
    mesh = torch.tt.init_device_mesh(mesh_2d_shape, mesh_dim_names=("rows", "cols"))
    axis_size = mesh.shape[axis]

    local = torch.randn(32, 32, dtype=torch.bfloat16)
    placements = [Partial(), Replicate()] if axis == 0 else [Replicate(), Partial()]
    dx = DTensor.from_local(local.to("tt"), mesh, placements, run_check=False)

    full = dx.full_tensor().cpu()
    torch.testing.assert_close(full, local * axis_size, atol=0.05, rtol=0.05)


@pytest.mark.xfail(
    reason="The topology fix removed the fabric routing crash (single-axis "
    "Shard(0)/Shard(1) now pass), but two-axis sharding still collapses one "
    "axis. DTensor performs a 2-axis shard as a sequence of single-axis "
    "scatters; our scatter_into rebuilds and *replaces* the whole multi-device "
    "storage per call, so the scatters don't compose — only the last mesh "
    "axis survives. The earlier axis's chunk is taken from rank 0's local "
    "tensor, which (single-coordinate fake PG, coordinate 0 on every axis) is "
    "always chunk 0 — so that axis ends up chunk-0-replicated. Needs the "
    "scatter seam to place chunks at full mesh coordinates.",
    strict=True,
)
def test_shard_both_axes(tt_pg, mesh_2d_shape) -> None:
    """`[Shard(0), Shard(1)]` — one shard per chip; `full_tensor()` gathers
    over both axes back to the global tensor."""
    mesh = torch.tt.init_device_mesh(mesh_2d_shape, mesh_dim_names=("rows", "cols"))
    rows, cols = mesh.shape

    x = torch.randn(32 * rows, 32 * cols, dtype=torch.bfloat16)
    dx = distribute_tensor(x.to("tt"), mesh, [Shard(0), Shard(1)])

    full = dx.full_tensor().cpu()
    torch.testing.assert_close(full, x, atol=0.05, rtol=0.05)


def test_same_ccl_across_meshes(tt_pg, mesh_2d_shape) -> None:
    """Compile-cache regression: the same TTIR (a 32x32 all-reduce) must not be
    replayed across meshes. A 1xN all-reduce compiles on the Ring fabric; the
    2-D long-axis all-reduce that follows hashes to the same module but needs
    Linear — a cache keyed on module hash alone replays Ring and fails."""
    n = torch.tt.num_chips()
    local = torch.randn(32, 32, dtype=torch.bfloat16)

    mesh_1d = torch.tt.init_device_mesh((n,), mesh_dim_names=("dp",))
    dx = DTensor.from_local(local.to("tt"), mesh_1d, [Partial()], run_check=False)
    torch.testing.assert_close(dx.full_tensor().cpu(), local * n, atol=0.05, rtol=0.05)

    mesh_2d = torch.tt.init_device_mesh(mesh_2d_shape, mesh_dim_names=("rows", "cols"))
    cols = mesh_2d.shape[1]
    dx = DTensor.from_local(local.to("tt"), mesh_2d, [Replicate(), Partial()], run_check=False)
    torch.testing.assert_close(dx.full_tensor().cpu(), local * cols, atol=0.05, rtol=0.05)
