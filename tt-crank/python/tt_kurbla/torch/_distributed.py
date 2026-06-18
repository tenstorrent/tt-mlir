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
        assert len(output_tensors) == 1, "tt.scatter: only single-output scatter supported"
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
