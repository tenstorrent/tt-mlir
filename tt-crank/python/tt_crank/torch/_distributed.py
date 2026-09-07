# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""tt c10d backend — single-process, fake-multi-rank.

Register a "tt" backend with `torch.distributed` so DTensor's collective
machinery dispatches to us when it sees a tt-device tensor. We claim
`world_size = num_chips()` from a single process; collectives translate into
operations on the underlying multi-device runtime tensor.

Implements all of the collectives that DTensor needs.

Importing this module registers the backend. Import order in __init__.py
matters: this must come after `_native` (which sets up the PrivateUse1
backend) and `_device.register()` (which renames PrivateUse1 to "tt").
"""

from __future__ import annotations

import functools

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import FakeWork

from . import _native


class TTProcessGroup(dist.ProcessGroup):
    """Single-process fake-multi-rank PG for the tt device.

    All "ranks" are this one process; collective methods translate into
    operations on the multi-device tt runtime tensor that backs the
    `at::Tensor` passed in.
    """

    def __init__(self, rank: int, world_size: int) -> None:
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size

    def size(self) -> int:
        return self._world_size

    def rank(self) -> int:
        return self._rank

    def getBackendName(self) -> str:
        return "tt"

    @property
    def pg_name(self) -> str:
        # Mirrors ProcessLocalGroup: read our name from the registered-PG
        # table _world maintains. PyTorch populates it when this PG is
        # registered via init_process_group / new_group.
        return dist.distributed_c10d._world.pg_names[self]

    @property
    def group_name(self) -> str:
        return self.pg_name

    def __repr__(self) -> str:
        return f"TTProcessGroup(world_size={self._world_size}, rank={self._rank})"

    @functools.cached_property
    def cluster_axis(self) -> int:
        """Runtime mesh axis this group's collectives span.

        Rank r sits at mesh coordinate (r // cols, r % cols) — torch lays mesh
        dims out row-major. A group along one mesh axis holds the other
        coordinate fixed, so the axis is whichever coordinate the group varies;
        it must vary it completely (one rank per coordinate value).

        Pure function of group composition, so it works for any group origin
        (`torch.tt.init_device_mesh`, upstream `init_device_mesh("tt", ...)`,
        `new_group`) and survives torch's process-group caching/reuse. Cached:
        the runtime mesh must not be reshaped under live groups.
        """
        rows, cols = _native.runtime_device_mesh_shape()
        ranks = dist.get_process_group_ranks(self)
        row_coords = {r // cols for r in ranks}
        col_coords = {r % cols for r in ranks}

        if len(row_coords) == rows and len(col_coords) == 1:
            return 0  # walks the rows of one column
        if len(col_coords) == cols and len(row_coords) == 1:
            return 1  # walks one row
        raise RuntimeError(
            f"tt collective group {ranks} does not span one axis of the {rows}x{cols} runtime mesh"
        )

    def broadcast(self, tensors, opts):
        """No-op for tt: tensors from `.to("tt")` are already replicated
        across the mesh (multi-device handle with the same data on every
        chip). Nothing to move.
        """
        assert len(tensors) >= 1, "tt.broadcast: expected at least one tensor"
        return FakeWork()

    def scatter(self, output_tensors, input_tensors, opts):
        """Scatter the source rank's chunks across this group's mesh axis.

        DTensor's `_shard_tensor` reaches here via `mesh_scatter` -> `dist.scatter`.
        `input_tensors` is a list of scatter-lists (one per output tensor, for
        coalesced scatter); we scatter a single tensor, so `input_tensors[0]` is
        the scatter-list: the global tensor split along the shard dim into one
        chunk per coordinate on the axis. `_native.scatter_into` places each chunk
        at its axis coordinate (every chip on the orthogonal axis takes its own
        shard) and replaces `output_tensors[0]`'s storage.
        """
        assert (
            len(output_tensors) == 1
        ), "tt.scatter: only single-output scatter supported"
        chunks = list(input_tensors[0]) if input_tensors else []
        _native.scatter_into(output_tensors[0], chunks, self.cluster_axis)
        return FakeWork()

    def _allgather_base(self, output, input, opts):
        """Gather every chip's slab into the full (replicated) `output`.

        Reads per-chip slabs from the multi-device handle — not `input.cpu()`,
        which returns shard 0 only.
        """
        _native.allgather_into(output, input, self.cluster_axis)
        return FakeWork()

    def allgather_into_tensor_coalesced(self, output_tensors, input_tensors, opts):
        """Functional-collective entry. Just iterates `_allgather_base`."""
        for o, i in zip(output_tensors, input_tensors, strict=True):
            self._allgather_base(o, i, opts)
        return FakeWork()

    def _reduce_scatter_base(self, output, input, opts):
        """Sum-reduce-scatter `input` into `output`, emitting `ttir.reduce_scatter`.
        The inverse of `_allgather_base`. This is the c10d entry torch drives, so
        it always scatters dim 0 (the `_reduce_scatter_base` contract). The
        Replicate -> Shard redistribute does not go through here — it calls
        `tt_crank.reduce_scatter` directly with the real shard dim (see
        `_install_replicate_to_shard_patch`).
        """
        if opts.reduceOp != dist.ReduceOp.SUM:
            raise NotImplementedError(
                f"tt reduce_scatter: only ReduceOp.SUM is supported, got {opts.reduceOp}"
            )
        _native.reduce_scatter_into(output, input, self.cluster_axis, 0)
        return FakeWork()

    def reduce_scatter_tensor_coalesced(self, output_tensors, input_tensors, opts):
        """Functional-collective entry. Just iterates `_reduce_scatter_base`."""
        for o, i in zip(output_tensors, input_tensors, strict=True):
            self._reduce_scatter_base(o, i, opts)
        return FakeWork()

    def allreduce(self, tensors, opts):
        """In-place elementwise sum over this group's mesh axis (emits
        `ttir.all_reduce`). DTensor's `Partial → Replicate` redistribute calls
        this. SUM only — the default Partial reduce_op.
        """
        if opts.reduceOp != dist.ReduceOp.SUM:
            raise NotImplementedError(
                f"tt allreduce: only ReduceOp.SUM is supported, got {opts.reduceOp}"
            )
        for t in tensors:
            _native.allreduce_into(t, self.cluster_axis)
        return FakeWork()


def _create_tt_pg(prefix_store, rank, world_size, timeout):
    """Backend factory invoked by `dist.init_process_group(backend="tt", ...)`."""
    return TTProcessGroup(rank, world_size)


# Old-style registration (extended_api=False / default) — the creator
# signature is (prefix_store, rank, world_size, timeout), which matches
# the pattern `multi_threaded_pg.py` uses for Python-subclass PGs.
dist.Backend.register_backend("tt", _create_tt_pg, devices=["tt", "cpu"])


def _reduce_scatter_out_shape(shape, group_size: int, scatter_dim: int) -> list[int]:
    """Output shape of a reduce-scatter: `scatter_dim` shrinks by `group_size`.

    Requires an even split (the only case the kernel handles); a clear error here
    beats a deeper TTIR verifier failure on an indivisible dim.
    """
    out = list(shape)
    dim = scatter_dim % len(out)
    if out[dim] % group_size:
        raise ValueError(
            f"tt_crank.reduce_scatter: dim {dim} (size {out[dim]}) is not "
            f"divisible by the group size {group_size}"
        )
    out[dim] //= group_size
    return out


# Collective that scatters the *real* shard dim. funcol's reduce_scatter_tensor
# only scatters dim 0 (faking other dims with a split we'd have to lower), so we
# expose our own op instead: TTIR/TTNN reduce_scatter carry an arbitrary
# scatter_dim, so we pass it straight through.
@torch.library.custom_op("tt_crank::reduce_scatter", mutates_args=())
def reduce_scatter(
    input: torch.Tensor, group_name: str, group_size: int, scatter_dim: int
) -> torch.Tensor:
    from torch.distributed.distributed_c10d import _resolve_process_group

    out_shape = _reduce_scatter_out_shape(input.shape, group_size, scatter_dim)
    output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    cluster_axis = _resolve_process_group(group_name).cluster_axis
    _native.reduce_scatter_into(output, input, cluster_axis, scatter_dim)
    return output


# Define the shape of the custom `reduce_scatter` op.
@reduce_scatter.register_fake
def _(input, group_name, group_size, scatter_dim):
    return input.new_empty(
        _reduce_scatter_out_shape(input.shape, group_size, scatter_dim)
    )


def _install_replicate_to_shard_patch() -> None:
    """Fix DTensor's Replicate->Shard redistribute for the single-process (rank) mesh.

    DTensor implements Replicate->Shard as a local rank operation (each rank
    already holds the whole tensor, so it can just keep its own chunk to
    create the shard). The tt backend drives the whole N-chip mesh from one
    fake rank whose coordinate is 0 (we are single-process) - so we cannot
    do the same, we need to create the whole tensor from the single process.

    Replace it with a reduce-scatter on the shard dim: the input is *replicated*,
    so a SUM reduce-scatter hands chip d `N * chunk_d`; dividing by N recovers
    `chunk_d`.

    Scoped to the tt mesh; every other backend keeps the stock local-chunk path.

    This patches a private DTensor method (`Shard._replicate_to_shard`), so it is
    coupled to that internal signature `(local_tensor, mesh, mesh_dim,
    shard_index)`.
    """
    from torch.distributed._functional_collectives import _resolve_group_name
    from torch.distributed.tensor.placement_types import Shard

    _orig = Shard._replicate_to_shard

    def _replicate_to_shard(self, local_tensor, mesh, mesh_dim, shard_index):
        if getattr(mesh, "device_type", None) != "tt":
            return _orig(self, local_tensor, mesh, mesh_dim, shard_index)
        num_chunks = mesh.size(mesh_dim)
        group_name = _resolve_group_name((mesh, mesh_dim))
        scattered = torch.ops.tt_crank.reduce_scatter(
            local_tensor, group_name, num_chunks, self.dim
        )
        return scattered / num_chunks

    Shard._replicate_to_shard = _replicate_to_shard


_install_replicate_to_shard_patch()
