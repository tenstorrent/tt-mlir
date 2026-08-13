# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pure, MLIR-free descriptor / shape math shared by both backends.

The device builder (`builder.py`) and the torch simulator (`sim/host.py`) must
agree on how a host op transforms a `Layout` -- otherwise the simulator, whose
whole job is to be an oracle for the device, could disagree on exactly the edge
cases it exists to check. These helpers compute the *output layout* (and, for
reshape, the resolved shape); each backend then does its own data movement
(device emits MLIR, sim does a torch roundtrip).

Kept free of `ttmlir` and `torch` imports so `import d2m_jit.sim` stays usable
with no tt-metal build (SIMULATOR_SPEC.md §1/§2).
"""

from .tensor_layout import Layout


def reduction_layout(layout: Layout, dim, allow_cross_tile: bool = False) -> Layout:
    """Return the output layout for a keepdim per-tile reduction.

    The DSL's float reductions can reduce across all tiles contained on one
    core. Reductions spanning multiple cores in the reduced dimension need a
    core gather/redistribute op to collect partials and place reduced values on
    the output-owning cores.
    """
    rank = len(layout.logical_shape)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise ValueError(
            f"reduce dim must be in range [-{rank}, {rank - 1}], got {dim}"
        )
    if layout.grid_shape[dim] > 1 and not allow_cross_tile:
        raise ValueError(
            "collapsed reductions only support a reduced logical dimension "
            "that fits on one core; got "
            f"{layout.grid_shape[dim]} cores along dimension {dim}. "
            "Pass allow_cross_tile=True only when the kernel has an explicit "
            "cross-core gather/redistribute strategy for the reduced dimension."
        )

    shape = list(layout.logical_shape)
    block_shape = list(layout.block_shape)
    grid_shape = list(layout.grid_shape)
    shape[dim] = 1
    block_shape[dim] = 1
    grid_shape[dim] = 1
    return layout.replace(shape=shape, block_shape=block_shape, grid_shape=grid_shape)


def resolve_reshape(layout: Layout, shape):
    """Resolve `reshape(lt, *shape)` args against a source `Layout`.

    `shape` is the `*shape` tuple as received by `reshape` -- either a single
    list/tuple or positional dims; a single `-1` dim is inferred from the rest.
    Returns `(new_shape, dst_layout)` (no data movement), raising `ValueError`
    on bad args or an element-count mismatch. The destination layout keeps the
    source `block_shape` / `grid_shape` when they still divide the new shape,
    else falls back to trivial `[1]*rank`.

    The actual reshape (host roundtrip) is left to each backend, since the
    source/result handles (`LazyTensor` vs `SimTensor`) differ.
    """
    # Accept reshape(lt, 1, 2, 256, 64) and reshape(lt, [1, 2, 256, 64]).
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        new_shape = tuple(shape[0])
    else:
        new_shape = tuple(shape)

    src_numel = 1
    for d in layout.logical_shape:
        src_numel *= d

    # Support the torch idiom of a single `-1` dim whose size is inferred
    # from the remaining dims (e.g. reshape(lt, -1) flattens; reshape(lt,
    # 2, -1) infers the second dim).
    neg_axes = [i for i, d in enumerate(new_shape) if d == -1]
    if len(neg_axes) > 1:
        raise ValueError(
            f"reshape: only one dimension may be inferred (-1), got {new_shape}"
        )
    if any(d < -1 for d in new_shape):
        raise ValueError(f"reshape: dimensions must be >= -1, got {new_shape}")
    if neg_axes:
        known = 1
        for d in new_shape:
            if d != -1:
                known *= d
        if known == 0 or src_numel % known != 0:
            raise ValueError(
                f"reshape: cannot infer -1 dimension: src has {src_numel} "
                f"elements which is not divisible by the product of the "
                f"known dims {known} (from {new_shape})"
            )
        inferred = src_numel // known
        new_shape = tuple(inferred if d == -1 else d for d in new_shape)

    dst_numel = 1
    for d in new_shape:
        dst_numel *= d
    if src_numel != dst_numel:
        raise ValueError(
            f"reshape: total element count must match: "
            f"src {tuple(layout.logical_shape)} ({src_numel}) "
            f"!= dst {new_shape} ({dst_numel})"
        )

    # Pick a destination layout: keep src's block/grid if compatible,
    # otherwise fall back to a trivial single-block single-grid layout
    # (the user can to_layout to something denser if perf matters).
    rank = len(new_shape)
    src_block = list(layout.block_shape)
    src_grid = list(layout.grid_shape)
    if (
        len(src_block) == rank
        and len(src_grid) == rank
        and all(
            d % (b * g * (32 if layout.tiled else 1)) == 0
            for d, b, g in zip(new_shape, src_block, src_grid)
        )
    ):
        block_shape = src_block
        grid_shape = src_grid
    else:
        block_shape = [1] * rank
        grid_shape = [1] * rank

    dst_layout = layout.replace(
        shape=new_shape,
        block_shape=block_shape,
        grid_shape=grid_shape,
    )
    return new_shape, dst_layout


# --- mesh sharding descriptor math ------------------------------------------
#
# Pure (MLIR-free, torch-free) helpers shared by the device builder and the
# simulator so both agree on how a mesh shard maps a full tensor to per-device
# shards. The device path emits `d2m.mesh_shard`; the sim path only tracks the
# metadata (multi-device data movement is not simulated).


class MeshShard:
    """Metadata needed to gather a per-device shard back to its full tensor."""

    __slots__ = ("full_shape", "shard_dims", "shard_shape")

    def __init__(self, full_shape, shard_dims, shard_shape):
        self.full_shape = list(full_shape)
        self.shard_dims = list(shard_dims)
        self.shard_shape = list(shard_shape)


def validate_mesh_decl(shape, topology):
    """Shared validation for a `mesh()` declaration (device builder and sim
    raise identical errors). Returns the normalized `(shape, topology)` as
    a list and a list-or-None."""
    shape = tuple(shape)
    if not shape or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape
    ):
        raise ValueError(f"mesh shape must contain positive integers, got {shape}")
    if len(shape) != 2:
        raise ValueError(f"TTMetal requires a rank-2 mesh, got shape {shape}")
    if topology is not None:
        topology = tuple(topology)
        if len(topology) != len(shape):
            raise ValueError(
                "mesh topology must have one entry per mesh dimension, "
                f"got shape {shape} and topology {topology}"
            )
        invalid = [
            value for value in topology if value not in {"disabled", "linear", "ring"}
        ]
        if invalid:
            raise ValueError(
                "mesh topology entries must be 'disabled', 'linear', or "
                f"'ring', got {invalid}"
            )
    return list(shape), list(topology) if topology is not None else None


def validate_mesh_mapping(mesh_shape, tensor_rank, shard_dims, shard_shape):
    if len(shard_dims) != len(mesh_shape):
        raise ValueError(
            "mesh shard_dims must have one entry per mesh dimension, "
            f"got mesh {mesh_shape} and shard_dims {shard_dims}"
        )
    if len(shard_shape) != tensor_rank:
        raise ValueError(
            "mesh shard_shape must have one entry per tensor dimension, "
            f"got tensor rank {tensor_rank} and shard_shape {shard_shape}"
        )

    for dim in shard_dims:
        if (
            not isinstance(dim, int)
            or isinstance(dim, bool)
            or dim < -1
            or dim >= tensor_rank
        ):
            raise ValueError(
                f"mesh shard dimension {dim!r} is invalid for rank {tensor_rank}"
            )
    for factor in shard_shape:
        if not isinstance(factor, int) or isinstance(factor, bool) or factor <= 0:
            raise ValueError(f"mesh shard factor must be positive, got {factor!r}")

    expected_shape = [1] * tensor_rank
    for mesh_axis, tensor_dim in enumerate(shard_dims):
        if tensor_dim >= 0:
            expected_shape[tensor_dim] *= mesh_shape[mesh_axis]
    if list(shard_shape) != expected_shape:
        raise ValueError(
            f"mesh shard_shape {shard_shape} does not match mesh {mesh_shape} "
            f"mapped by shard_dims {shard_dims}; expected {expected_shape}"
        )


def shard_logical_shape(mesh_shape, full_shape, shard_dims, shard_shape):
    validate_mesh_mapping(mesh_shape, len(full_shape), shard_dims, shard_shape)
    shard = list(full_shape)
    for dim, factor in enumerate(shard_shape):
        if shard[dim] % factor != 0:
            raise ValueError(
                f"mesh shard: full dim {dim} ({shard[dim]}) is not divisible "
                f"by shard factor {factor}"
            )
        shard[dim] //= factor
    return shard


# --- current mesh mirror -----------------------------------------------------
#
# The device builder is the source of truth for the declared mesh (it owns the
# `ttcore.meshes` module attribute), but it lives behind the MLIR bindings. The
# builder mirrors the declaration here so the simulator can validate mesh ops
# without importing the device builder (keeping `import d2m_jit.sim` device-free
# per SIMULATOR_SPEC.md). There is a single builder singleton, so a single
# mirror suffices; the builder clears it when it starts a fresh graph.

_CURRENT_MESH = None


def set_current_mesh(shape, topology, name):
    global _CURRENT_MESH
    _CURRENT_MESH = {
        "shape": list(shape),
        "topology": list(topology) if topology is not None else None,
        "name": name,
    }


def clear_current_mesh():
    global _CURRENT_MESH
    _CURRENT_MESH = None


def current_mesh():
    """Return the declared mesh mirror ({"shape", "topology", "name"}) or None."""
    return _CURRENT_MESH
