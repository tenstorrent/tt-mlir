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

import torch

from ..tensor_layout import Layout
from .tensors import SimTensor, tile_padded_shape, torch_dtype


def _alloc(layout: Layout, fill=None):
    shape = tile_padded_shape(layout)
    dtype = torch_dtype(layout)
    if fill is None or fill == 0:
        return torch.zeros(shape, dtype=dtype)
    return torch.full(shape, fill, dtype=dtype)


def to_layout(input_, layout: Layout) -> SimTensor:
    """Bring a torch tensor onto the (simulated) device, or re-layout an
    existing SimTensor. Materialises views (`.contiguous()` semantics)."""
    if isinstance(input_, SimTensor):
        assert list(input_.layout.logical_shape) == list(layout.logical_shape), (
            f"to_layout shape mismatch: src {input_.layout.logical_shape} "
            f"vs target {layout.logical_shape}"
        )
        logical = input_.to_logical()
        buf = _alloc(layout)
        r, c = layout.logical_shape
        buf[:r, :c] = logical.to(buf.dtype)
        return SimTensor(layout, buf, is_view=False)

    if isinstance(input_, torch.Tensor):
        assert list(input_.shape) == list(layout.logical_shape), (
            f"to_layout shape mismatch: tensor {list(input_.shape)} "
            f"vs layout {layout.logical_shape}"
        )
        buf = _alloc(layout)
        r, c = layout.logical_shape
        buf[:r, :c] = input_.to(buf.dtype)
        return SimTensor(layout, buf, is_view=False)

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
    buf = lt.buffer.permute(*perm)
    new_layout = lt.layout.replace(
        shape=[lt.layout.logical_shape[p] for p in perm],
        block_shape=[lt.layout.block_shape[p] for p in perm],
        grid_shape=[lt.layout.grid_shape[p] for p in perm],
    )
    return SimTensor(new_layout, buf, is_view=True)


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


# --- reductions / materialisation -------------------------------------------


def reduction_layout(layout: Layout, dim, allow_cross_tile: bool = False) -> Layout:
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
            f"that fits on one core; got {layout.grid_shape[dim]} cores along "
            f"dimension {dim}. Pass allow_cross_tile=True only when the kernel "
            "has an explicit cross-core gather/redistribute strategy."
        )
    shape = list(layout.logical_shape)
    block_shape = list(layout.block_shape)
    grid_shape = list(layout.grid_shape)
    shape[dim] = 1
    block_shape[dim] = 1
    grid_shape[dim] = 1
    return layout.replace(shape=shape, block_shape=block_shape, grid_shape=grid_shape)


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
