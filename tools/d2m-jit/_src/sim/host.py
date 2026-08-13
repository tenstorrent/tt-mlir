# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side orchestration ops for the simulator (the sim analogs of the
constructors / materialisers in `_src/builder.py`).

A `SimTensor` only ever stores the *logical* data in a tile-padded torch
buffer: tiling, blocked vs. user grid, mem_space, and collapse carry no
information that changes output values, so they are descriptor-only here.
"""

import inspect
import itertools

import torch

from ..tensor_layout import Layout
from ..layout_math import (
    reduction_layout,
    resolve_reshape,
    MeshShard,
    validate_mesh_decl,
    validate_mesh_mapping,
    shard_logical_shape,
    current_mesh,
    set_current_mesh,
)
from .tensors import SimTensor, tile_padded_shape, torch_dtype


def _num_devices():
    """Device count of the declared mesh (1 with no mesh declared)."""
    mesh = current_mesh()
    if mesh is None:
        return 1
    n = 1
    for dim in mesh["shape"]:
        n *= dim
    return n


def _alloc_one(layout: Layout, fill=None):
    shape = tile_padded_shape(layout)
    dtype = torch_dtype(layout)
    if fill is None or fill == 0:
        return torch.zeros(shape, dtype=dtype)
    return torch.full(shape, fill, dtype=dtype)


def _alloc(layout: Layout, fill=None):
    """One tile-padded buffer per mesh device, replicated fill -- under a mesh
    every tensor exists once per chip (SIMULATOR_SPEC.md §14.1). A single
    buffer when no mesh is declared."""
    return [_alloc_one(layout, fill) for _ in range(_num_devices())]


def to_layout(input_, layout: Layout) -> SimTensor:
    """Bring a torch tensor onto the (simulated) device, or re-layout an
    existing SimTensor. Materialises views (`.contiguous()` semantics)."""
    if isinstance(input_, SimTensor):
        assert list(input_.layout.logical_shape) == list(layout.logical_shape), (
            f"to_layout shape mismatch: src {input_.layout.logical_shape} "
            f"vs target {layout.logical_shape}"
        )
        # Re-layout is per-device: each of the source's buffers materialises
        # into a fresh buffer (SIMULATOR_SPEC.md §14.1).
        r, c = layout.logical_shape
        bufs = []
        for src in input_.buffers:
            buf = _alloc_one(layout)
            buf[:r, :c] = src[:r, :c].to(buf.dtype)
            bufs.append(buf)
        return SimTensor(layout, bufs, is_view=False)

    if isinstance(input_, torch.Tensor):
        assert list(input_.shape) == list(layout.logical_shape), (
            f"to_layout shape mismatch: tensor {list(input_.shape)} "
            f"vs layout {layout.logical_shape}"
        )
        r, c = layout.logical_shape
        bufs = _alloc(layout)
        for buf in bufs:
            buf[:r, :c] = input_.to(buf.dtype)
        return SimTensor(layout, bufs, is_view=False)

    raise TypeError(
        f"to_layout expected a torch.Tensor or SimTensor, got "
        f"{type(input_).__name__}"
    )


def empty(layout: Layout) -> SimTensor:
    # Device `empty` is undefined; sim uses zeros so results are deterministic
    # (documented divergence -- see SIMULATOR_SPEC.md §9).
    return SimTensor(layout, _alloc(layout))


def zeros(layout: Layout) -> SimTensor:
    return SimTensor(layout, _alloc(layout, 0))


def full(layout: Layout, value) -> SimTensor:
    return SimTensor(layout, _alloc(layout, value))


def tilize(lt: SimTensor, dtype=None) -> SimTensor:
    if not isinstance(lt, SimTensor):
        raise TypeError(f"tilize expected a SimTensor, got {type(lt).__name__}")
    overrides = {"tiled": True}
    if dtype is not None:
        overrides["dtype"] = dtype
    return to_layout(lt, lt.layout.replace(**overrides))


def untilize(lt: SimTensor, dtype=None) -> SimTensor:
    if not isinstance(lt, SimTensor):
        raise TypeError(f"untilize expected a SimTensor, got {type(lt).__name__}")
    overrides = {"tiled": False}
    if dtype is not None:
        overrides["dtype"] = dtype
    return to_layout(lt, lt.layout.replace(**overrides))


# --- host constructors / reshape --------------------------------------------
#
# Sim mirrors of the host constructors in `_src/builder.py`. Both back the same
# `torch.arange`/`torch.reshape` semantics the device path lowers to (host
# roundtrip + to_layout). Physical tiling/grid carry no values in the sim
# (SIMULATOR_SPEC.md §3), so only the logical fill/shape matters here.


def arange(layout: Layout, start: int = 0, step: int = 1) -> SimTensor:
    """Sim analog of `builder.arange`: row-major `torch.arange` over
    `layout.logical_shape`, then `to_layout`."""
    dtype = torch_dtype(layout)
    numel = 1
    for d in layout.logical_shape:
        numel *= d
    flat = torch.arange(start, start + numel * step, step, dtype=dtype)
    return to_layout(flat.reshape(list(layout.logical_shape)), layout)


def reshape(lt: SimTensor, *shape) -> SimTensor:
    """Sim analog of `builder.reshape`: a host roundtrip (`to_host` ->
    `torch.reshape` -> `to_layout`). Shape resolution is the shared
    `resolve_reshape`, so the resolved layout and the error messages the
    negative tests assert are identical to the device path."""
    if not isinstance(lt, SimTensor):
        raise TypeError(f"reshape expected a SimTensor, got {type(lt).__name__}")

    new_shape, dst_layout = resolve_reshape(lt.layout, shape)
    if lt.is_view:
        # Raises the canonical to_host view rejection -- the exact error the
        # device path hits, since its reshape starts with a to_host roundtrip.
        to_host(lt)
    # The roundtrip is per-device (SIMULATOR_SPEC.md §14.1).
    r, c = lt.layout.logical_shape
    bufs = []
    for src in lt.buffers:
        buf = _alloc_one(dst_layout)
        dr, dc = dst_layout.logical_shape
        buf[:dr, :dc] = src[:r, :c].reshape(new_shape).to(buf.dtype)
        bufs.append(buf)
    return SimTensor(dst_layout, bufs)


def spatial(inputs, outputs, grid_ranges, region_builders):
    """Sim analog of `builder.spatial`.

    On device this emits a `d2m.spatial` op and lays each region's kernel onto
    a distinct core range. Physical placement carries no values in the sim
    (SIMULATOR_SPEC.md §3), so `grid_ranges` is only length-checked, not
    applied: running each region builder is enough. A sim kernel call executes
    eagerly and mutates its output SimTensors in place (via `remote_store`), so
    after every region has run the outputs already hold their results.

    `core_index` inside a region is local to that region's `grid=` (0-based),
    exactly as on device -- the region's grid offset is a placement detail, not
    part of the logical block addressing the kernels use.
    """
    builders = list(region_builders)
    ranges = list(grid_ranges)
    if len(ranges) != len(builders):
        raise ValueError(
            f"grid_ranges has {len(ranges)} entries but region_builders has "
            f"{len(builders)}"
        )
    if not builders:
        raise ValueError("d2m.spatial requires at least one region")
    for i, builder_fn in enumerate(builders):
        if not callable(builder_fn):
            raise TypeError(
                f"region_builders[{i}] must be callable, got "
                f"{type(builder_fn).__name__}"
            )

    for i, v in enumerate(inputs):
        if not isinstance(v, SimTensor):
            raise TypeError(
                f"d2m.spatial inputs[{i}] is {type(v).__name__}, expected " f"SimTensor"
            )
    output_lts = list(outputs)
    if not output_lts:
        raise ValueError("d2m.spatial outputs= must be non-empty")
    for i, v in enumerate(output_lts):
        if not isinstance(v, SimTensor):
            raise TypeError(
                f"d2m.spatial outputs[{i}] is {type(v).__name__}, expected "
                f"SimTensor"
            )

    # The region builders close over their input/output SimTensors and call a
    # sim kernel, which runs immediately and writes into the outputs. The
    # device's post-hoc check that the emitted region kernels wrote exactly
    # outputs= is an MLIR-graph invariant with no value analog here.
    for build_region in builders:
        build_region()
    return tuple(output_lts)


# --- views -------------------------------------------------------------------


class _Dim:
    def __init__(self, pos):
        self.pos = pos


def _lambda_spec(fn):
    """Run `fn` with sentinel dims; return (n_params, spec) where spec is a
    list of ("dim", pos) / ("const", 0) per result."""
    n = len(inspect.signature(fn).parameters)
    results = fn(*[_Dim(i) for i in range(n)])
    spec = []
    for r in results:
        if isinstance(r, _Dim):
            spec.append(("dim", r.pos))
        elif isinstance(r, int) and r == 0:
            spec.append(("const", 0))
        else:
            raise TypeError(f"unsupported view result {r!r}")
    return n, spec


def _apply_perm(lt: SimTensor, perm) -> SimTensor:
    n = len(lt.layout.logical_shape)
    if sorted(perm) != list(range(n)):
        raise ValueError(
            f"permutation {list(perm)} is not a rearrangement of (0..{n - 1})"
        )
    if n != 2:
        raise NotImplementedError("sim views support rank-2 tensors only")
    bufs = [buf.permute(*perm) for buf in lt.buffers]
    new_layout = lt.layout.replace(
        shape=[lt.layout.logical_shape[p] for p in perm],
        block_shape=[lt.layout.block_shape[p] for p in perm],
        grid_shape=[lt.layout.grid_shape[p] for p in perm],
    )
    return SimTensor(new_layout, bufs, is_view=True)


def permute(lt: SimTensor, *dims) -> SimTensor:
    if not isinstance(lt, SimTensor):
        raise TypeError(f"permute expected a SimTensor, got {type(lt).__name__}")
    n = len(lt.layout.logical_shape)
    if len(dims) != n:
        raise ValueError(
            f"permute: expected {n} dim indices for logical rank {n}, "
            f"got {len(dims)}: {dims}"
        )
    return _apply_perm(lt, list(dims))


def view(lt: SimTensor, remapping_fn) -> SimTensor:
    if not isinstance(lt, SimTensor):
        raise TypeError(f"view expected a SimTensor, got {type(lt).__name__}")
    n = len(lt.layout.logical_shape)
    nparams, spec = _lambda_spec(remapping_fn)
    if nparams != n or any(tag != "dim" for tag, _ in spec) or len(spec) != n:
        raise ValueError(
            "view: lambda must be a permutation of logical dims (no constants); "
            "use view_layout for richer remappings"
        )
    return _apply_perm(lt, [pos for _, pos in spec])


def view_layout(lt: SimTensor, remapping_fn) -> SimTensor:
    if not isinstance(lt, SimTensor):
        raise TypeError(f"view_layout expected a SimTensor, got {type(lt).__name__}")
    n = len(lt.layout.logical_shape)
    nparams, spec = _lambda_spec(remapping_fn)
    if nparams != 2 * n:
        raise ValueError(
            f"view_layout: lambda takes {nparams} args but source MLIR rank "
            f"is {2 * n}"
        )
    head = spec[:n]
    tail = spec[n:]
    if any(tag != "dim" for tag, _ in head):
        raise NotImplementedError(
            "sim view_layout supports paired (grid, tile) permutations only "
            "(broadcast/const remaps are not modeled yet)"
        )
    perm = [pos for _, pos in head]
    for i, (tag, pos) in enumerate(tail):
        if tag != "dim" or pos != perm[i] + n:
            raise NotImplementedError(
                "sim view_layout supports paired (grid, tile) permutations only"
            )
    return _apply_perm(lt, perm)


# --- mesh --------------------------------------------------------------------
#
# The sim's mesh state is the `layout_math` mirror the device builder also
# writes to -- `mesh()` here only records the declaration (no MLIR, no builder
# scope; SIMULATOR_SPEC.md §14.2). `mesh_shard` performs the real full->shards
# split (§14.3). The gather direction is not simulated yet (§14.5, #9202):
# `mesh_gather` only derives/validates the gather metadata -- exactly the pure
# descriptor math the device path runs -- so `.mesh.full_shape` matches the
# device path.


def mesh(shape, topology=None):
    """Sim analog of `builder.mesh`: declare the device mesh.

    Validation is the shared `validate_mesh_decl`, so the errors match the
    device path. `topology` is recorded but value-neutral -- the sim models no
    fabric (SIMULATOR_SPEC.md §14.7). Lifecycle divergence (§14.2): the sim has
    no graph lifecycle, so redeclaration simply replaces the current mesh;
    consistency is enforced where it matters, at `mesh_shard` / `mesh_gather`
    validation time.
    """
    shape, topology = validate_mesh_decl(shape, topology)
    set_current_mesh(shape, topology, "mesh")


def mesh_shard(input_, layout: Layout, shard_dims, shard_shape) -> SimTensor:
    """Sim analog of `builder.mesh_shard`: distribute a full host tensor into
    one `layout` shard per device.

    Shapes and errors are the shared descriptor math (`shard_logical_shape`),
    so they match the device path. Chunk placement matches the runtime
    (`runtime/lib/ttmetal/meshshard_utils.cpp::shard`): walking mesh axes in
    order, each non-`-1` entry of `shard_dims` splits its tensor dim evenly
    across that mesh axis (so chunk indices run last-mesh-axis-fastest, the
    runtime's `incrementIndices` order), and replicated (`-1`) axes do not
    narrow -- every device along them gets a copy of the same chunk, matching
    the runtime's replicate fill (SIMULATOR_SPEC.md §14.3).
    """
    if not isinstance(input_, torch.Tensor):
        raise TypeError("mesh_shard expects a torch.Tensor containing the full tensor")
    mesh = current_mesh()
    if mesh is None:
        raise RuntimeError("mesh_shard() requires a preceding mesh() declaration")

    shard_dims = list(shard_dims)
    shard_shape = list(shard_shape)
    full_shape = list(input_.shape)
    mesh_shape = mesh["shape"]
    expected_shape = shard_logical_shape(
        mesh_shape, full_shape, shard_dims, shard_shape
    )
    if list(layout.logical_shape) != expected_shape:
        raise ValueError(
            f"mesh_shard layout shape {list(layout.logical_shape)} does not "
            f"match expected per-device shape {expected_shape}"
        )

    bufs = []
    for coord in itertools.product(*[range(n) for n in mesh_shape]):
        chunk = input_
        for axis, m in enumerate(coord):
            dim = shard_dims[axis]
            if dim < 0:
                continue
            # Divisibility of the running extent is implied by
            # shard_logical_shape's full-dim divisibility check.
            extent = chunk.shape[dim] // mesh_shape[axis]
            chunk = chunk.narrow(dim, m * extent, extent)
        buf = _alloc_one(layout)
        r, c = expected_shape
        buf[:r, :c] = chunk.to(buf.dtype)
        bufs.append(buf)
    return SimTensor(layout, bufs, mesh=MeshShard(full_shape, shard_dims, shard_shape))


def mesh_gather(lt: SimTensor, shard_dims=None, shard_shape=None) -> SimTensor:
    """Sim analog of `builder.mesh_gather`: mark a per-device SimTensor for a
    `shard_to_full` gather and record the resulting full-tensor metadata.

    A SimTensor is already materialised (no lazy `_resolve()` as on device), so
    this only attaches the `MeshShard` mapping; the shard math is the shared
    `validate_mesh_mapping`, so the derived `full_shape` and the errors the
    negative tests assert match the device path.
    """
    if not isinstance(lt, SimTensor):
        raise TypeError(f"mesh_gather expected a SimTensor, got {type(lt).__name__}")
    mesh = current_mesh()
    if mesh is None:
        raise RuntimeError("mesh_gather() requires a preceding mesh() declaration")

    if lt.mesh is not None:
        if shard_dims is not None and list(shard_dims) != lt.mesh.shard_dims:
            raise ValueError("mesh_gather shard_dims do not match existing metadata")
        if shard_shape is not None and list(shard_shape) != lt.mesh.shard_shape:
            raise ValueError("mesh_gather shard_shape does not match existing metadata")
        return lt
    if shard_dims is None or shard_shape is None:
        raise ValueError(
            "mesh_gather needs shard_dims and shard_shape for a tensor not "
            "produced by mesh_shard"
        )

    shard_dims = list(shard_dims)
    shard_shape = list(shard_shape)
    validate_mesh_mapping(
        mesh["shape"], len(lt.layout.logical_shape), shard_dims, shard_shape
    )
    full_shape = [
        dim * factor for dim, factor in zip(lt.layout.logical_shape, shard_shape)
    ]
    lt.mesh = MeshShard(full_shape, shard_dims, shard_shape)
    return lt


# --- reductions / materialisation -------------------------------------------
#
# `reduction_layout` is the shared pure-descriptor helper, imported above and
# re-exported here (and thus through the sim package) so the sim surface still
# offers it.


def to_host(*lts: SimTensor):
    if not lts:
        raise ValueError("to_host requires at least one SimTensor")
    for i, lt in enumerate(lts):
        if not isinstance(lt, SimTensor):
            raise TypeError(
                f"to_host argument {i} is {type(lt).__name__}, expected SimTensor"
            )
        if lt.is_view:
            raise ValueError(
                f"to_host: argument {i} is a view (created via "
                f"view/view_layout/permute). Views are metadata "
                f"reinterpretations and cannot be materialised directly. "
                f"Convert first, e.g. to_layout(v, v.layout)."
            )
    return tuple(lt.to_logical() for lt in lts)
