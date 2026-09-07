# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Replicate -> Shard redistribute on the tt mesh.
"""

import pytest
import torch
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)

pytestmark = pytest.mark.multichip


def _distinct(rows: int, cols: int) -> torch.Tensor:
    """A tensor whose every chunk differs."""
    return torch.arange(rows * cols, dtype=torch.bfloat16).reshape(rows, cols)


@pytest.mark.parametrize("dim", [0, 1])
def test_replicate_to_shard_eager(tt_pg, dim: int) -> None:
    """Eager Replicate -> Shard(dim) -> Replicate must round-trip to the
    original. Dispatches through the 'tt_kurbla.reduce_scatter' op."""
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("x",))
    x = _distinct(n, 3) if dim == 0 else _distinct(3, n)

    replicated = distribute_tensor(x.to("tt"), mesh, [Replicate()])
    sharded = replicated.redistribute(mesh, [Shard(dim)])
    back = sharded.redistribute(mesh, [Replicate()]).to_local().cpu()

    torch.testing.assert_close(back, x, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dim", [0, 1])
def test_replicate_to_shard_compile(tt_pg, dim: int) -> None:
    """The Replicate -> Shard(dim) redistribute traced through the tt compile
    backend, where it lowers to 'ttir.reduce_scatter'."""
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("x",))
    x = _distinct(n, 3) if dim == 0 else _distinct(3, n)

    def shard(dt: torch.Tensor):
        return dt.redistribute(mesh, [Shard(dim)])

    replicated = distribute_tensor(x.to("tt"), mesh, [Replicate()])
    sharded = torch.compile(shard, backend="tt", dynamic=False)(replicated)
    back = sharded.full_tensor().cpu()

    torch.testing.assert_close(back, x, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("axis", [0, 1])
def test_replicate_to_shard_2d(tt_pg, mesh_2d_shape, axis: int) -> None:
    """Replicate -> Shard(0) on one axis of a 2-D mesh (the other axis stays
    replicated), then back. Exercises the patch composing per mesh-dim: the
    reduce-scatter must run over the chosen axis only."""
    mesh = torch.tt.init_device_mesh(mesh_2d_shape, mesh_dim_names=("rows", "cols"))
    axis_size = mesh.shape[axis]
    x = _distinct(axis_size * 2, 4)  # tensor dim 0 must divide the sharding axis

    replicated = [Replicate(), Replicate()]
    target = [Replicate(), Replicate()]
    target[axis] = Shard(0)
    d = distribute_tensor(x.to("tt"), mesh, replicated)
    back = d.redistribute(mesh, target).redistribute(mesh, replicated).to_local().cpu()

    torch.testing.assert_close(back, x, atol=0.0, rtol=0.0)


def test_reduce_scatter_pg(tt_pg) -> None:
    """The c10d reduce-scatter path that torch drives through
    'TTProcessGroup._reduce_scatter_base' (scatter dim 0). A Partial -> Shard(0)
    redistribute reduce-scatters: every chip contributes the same local 'x', so
    the summed-then-scattered result gathers back to 'n * x'."""
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("x",))
    x = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4)

    partial = DTensor.from_local(x.to("tt"), mesh, [Partial()], run_check=False)
    sharded = partial.redistribute(mesh, [Shard(0)])  # reduce-scatter (sum)
    full = sharded.full_tensor().cpu()  # all-gather

    torch.testing.assert_close(full, x * n, atol=0.0, rtol=0.0)
