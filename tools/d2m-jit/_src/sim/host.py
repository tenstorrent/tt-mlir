# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simulator host API: the sim analogues of the `_src/builder.py` surface.

These are bound over the real builder functions in `api.py` when
`config.simulator` is set. Kernels run eagerly (each call mutates its output
`SimTensor` in place); `to_host` just crops the physical buffer back to the
logical shape.
"""

from __future__ import annotations

import itertools
import math
import types

import torch

from .tensor import SimTensor, torch_dtype
from .runtime import run_kernel

# --- LazyTensor -------------------------------------------------------------

_generation = [1]


class SimLazyTensor:
    """Host handle wrapping a mutable `SimTensor` (or a materialised result)."""

    __slots__ = ("layout", "_sim", "materialized", "is_view", "generation")

    def __init__(self, layout, sim_tensor, is_view=False):
        self.layout = layout
        self._sim = sim_tensor
        self.materialized = None
        self.is_view = is_view
        self.generation = _generation[0]

    def _ensure_fresh(self):
        # Mirror the real builder: a tensor from a prior generation that was
        # not materialised by its to_host() is "spent" and must not be reused.
        # Materialised tensors auto-re-enter (see _sim_resolve).
        if self.generation != _generation[0] and self.materialized is None:
            raise RuntimeError(
                "Stale LazyTensor: produced by a prior builder generation that "
                "was reset by to_host(). Re-materialise its source or include it "
                "in the to_host() call before reset."
            )

    def _sim_resolve(self) -> SimTensor:
        self._ensure_fresh()
        if self._sim is not None:
            return self._sim
        if self.materialized is not None:
            self._sim = SimTensor.from_torch(self.layout, self.materialized)
            return self._sim
        raise RuntimeError("SimLazyTensor has no backing data")

    def _sim_logical(self) -> torch.Tensor:
        self._ensure_fresh()
        if self._sim is not None:
            return self._sim.to_logical()
        if self.materialized is not None:
            return self.materialized
        raise RuntimeError("SimLazyTensor has no backing data")

    @property
    def value(self):
        # Compatibility shim for host code that inspects the MLIR value's
        # physical (blocked) shape, e.g. rope's half-roll builds its remap from
        # `lt.value.type.shape`. The device tile-grid shape is
        # [grid dims..., block dims...] (block extents in tiles for a tiled
        # layout, in elements otherwise).
        layout = self.layout
        if layout.tiled:
            phys = list(layout.grid_shape) + list(layout.block_shape)
        else:
            phys = [g * b for g, b in zip(layout.grid_shape, layout.block_shape)]
        return _SimValueShim(_SimTypeShim(phys))

    def to_host(self):
        return to_host(self)[0]


class _SimTypeShim:
    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = shape


class _SimValueShim:
    __slots__ = ("type",)

    def __init__(self, type_):
        self.type = type_


# --- constructors -----------------------------------------------------------


def to_layout(input_, layout) -> SimLazyTensor:
    if isinstance(input_, SimLazyTensor):
        assert list(input_.layout.logical_shape) == list(layout.logical_shape), (
            f"to_layout shape mismatch: src {input_.layout.logical_shape} "
            f"vs target {layout.logical_shape}"
        )
        logical = input_._sim_logical().to(torch_dtype(layout.dtype))
        return SimLazyTensor(layout, SimTensor.from_torch(layout, logical))

    if isinstance(input_, torch.Tensor):
        assert list(input_.shape) == list(layout.logical_shape), (
            f"to_layout shape mismatch: tensor {list(input_.shape)} "
            f"vs layout {layout.logical_shape}"
        )
        return SimLazyTensor(layout, SimTensor.from_torch(layout, input_))

    raise TypeError(
        f"to_layout expected a torch.Tensor or LazyTensor, got {type(input_).__name__}"
    )


def tilize(lt, dtype=None) -> SimLazyTensor:
    if not isinstance(lt, SimLazyTensor):
        raise TypeError(f"tilize expected a LazyTensor, got {type(lt).__name__}")
    overrides = {"tiled": True}
    if dtype is not None:
        overrides["dtype"] = dtype
    return to_layout(lt, lt.layout.replace(**overrides))


def untilize(lt, dtype=None) -> SimLazyTensor:
    if not isinstance(lt, SimLazyTensor):
        raise TypeError(f"untilize expected a LazyTensor, got {type(lt).__name__}")
    overrides = {"tiled": False}
    if dtype is not None:
        overrides["dtype"] = dtype
    return to_layout(lt, lt.layout.replace(**overrides))


def empty(layout) -> SimLazyTensor:
    return SimLazyTensor(layout, SimTensor.empty(layout))


def full(layout, value) -> SimLazyTensor:
    t = torch.full(
        list(layout.logical_shape), float(value), dtype=torch_dtype(layout.dtype)
    )
    return SimLazyTensor(layout, SimTensor.from_torch(layout, t))


def zeros(layout) -> SimLazyTensor:
    return full(layout, 0)


def arange(layout, start: int = 0, step: int = 1) -> SimLazyTensor:
    numel = math.prod(layout.logical_shape)
    flat = torch.arange(
        start, start + numel * step, step, dtype=torch_dtype(layout.dtype)
    )
    return to_layout(flat.reshape(list(layout.logical_shape)), layout)


# --- views ------------------------------------------------------------------


def _as_perm_result(fn, n):
    """Call `fn` with ints 0..n-1 and return the resulting tuple as ints."""
    try:
        result = fn(*range(n))
    except TypeError as e:
        raise ValueError(f"view lambda arity does not match rank {n}: {e}")
    return list(result)


def permute(lt, *dims) -> SimLazyTensor:
    if not isinstance(lt, SimLazyTensor):
        raise TypeError(f"permute expected a LazyTensor, got {type(lt).__name__}")
    n = len(lt.layout.logical_shape)
    if len(dims) != n:
        raise ValueError(
            f"permute: expected {n} dim indices for logical rank {n}, got {len(dims)}"
        )
    if sorted(dims) != list(range(n)):
        raise ValueError(f"permute: {list(dims)} is not a permutation of 0..{n - 1}")
    logical = lt._sim_logical().permute(*dims).contiguous()
    new_layout = lt.layout.replace(
        shape=[lt.layout.logical_shape[d] for d in dims],
        block_shape=[lt.layout.block_shape[d] for d in dims],
        grid_shape=[lt.layout.grid_shape[d] for d in dims],
    )
    return SimLazyTensor(
        new_layout, SimTensor.from_torch(new_layout, logical), is_view=True
    )


def view(lt, remapping_fn) -> SimLazyTensor:
    if not isinstance(lt, SimLazyTensor):
        raise TypeError(f"view expected a LazyTensor, got {type(lt).__name__}")
    n = len(lt.layout.logical_shape)
    spec = _as_perm_result(remapping_fn, n)
    if sorted(spec) != list(range(n)) or any(not isinstance(v, int) for v in spec):
        raise ValueError(
            "view: lambda must be a permutation of logical dims (no constants); "
            "use view_layout for richer remappings"
        )
    return permute(lt, *spec)


def view_layout(lt, remapping_fn) -> SimLazyTensor:
    if not isinstance(lt, SimLazyTensor):
        raise TypeError(f"view_layout expected a LazyTensor, got {type(lt).__name__}")
    n = len(lt.layout.logical_shape)
    phys_rank = 2 * n if lt.layout.tiled else n
    spec = _as_perm_result(remapping_fn, phys_rank)

    if spec == list(range(phys_rank)):
        # Identity: metadata-only, data unchanged.
        logical = lt._sim_logical()
        return SimLazyTensor(
            lt.layout, SimTensor.from_torch(lt.layout, logical), is_view=True
        )

    # Paired (grid, tile) permutation: [p0..p_{n-1}, p0+n..p_{n-1}+n].
    head, tail = spec[:n], spec[n:]
    if (
        all(isinstance(v, int) and v < n for v in head)
        and sorted(head) == list(range(n))
        and tail == [p + n for p in head]
    ):
        return permute(lt, *head)

    # Affine-arithmetic remapping (e.g. the rope half-roll): evaluate the lambda
    # concretely at every physical index and gather the source tile into the
    # destination position. The lambda maps a destination index tuple to its
    # source index tuple (dst[I] = src[fn(*I)]), matching the MLIR view_layout
    # affine map's dst->src direction. Arithmetic maps preserve the source
    # physical shape and inherit the source Layout unchanged (SIMULATOR_SPEC §8).
    return _arithmetic_view_layout(lt, remapping_fn, phys_rank)


def _eval_remap(remapping_fn, dst_idx, phys_rank, idx_bounds):
    """Evaluate the dst->src remapping lambda at one destination index and
    validate the result: right arity, integer coordinates, all in bounds."""
    src_idx = tuple(remapping_fn(*dst_idx))
    if len(src_idx) != phys_rank:
        raise ValueError(
            f"view_layout lambda returned {len(src_idx)} indices for physical "
            f"rank {phys_rank}"
        )
    for k, (s, bound) in enumerate(zip(src_idx, idx_bounds)):
        if not isinstance(s, int) or isinstance(s, bool):
            raise ValueError(
                f"view_layout lambda must return integer indices; got {s!r} "
                f"at position {k}"
            )
        if not (0 <= s < bound):
            raise ValueError(
                f"view_layout source index {src_idx} out of bounds at "
                f"position {k} (extent {bound})"
            )
    return src_idx


def _arithmetic_view_layout(lt, remapping_fn, phys_rank) -> SimLazyTensor:
    layout = lt.layout
    n = len(layout.logical_shape)
    src_phys = lt._sim_resolve().physical

    if not layout.tiled:
        # Non-tiled: the physical buffer already *is* the element-space affine
        # index grid (physical rank == n, one index per logical axis with extent
        # grid[a]*block[a] == logical[a]). The lambda addresses individual
        # elements, so we gather element-by-element with no tile split.
        idx_bounds = [g * b for g, b in zip(layout.grid_shape, layout.block_shape)]
        dst = torch.empty_like(src_phys)
        for dst_idx in itertools.product(*(range(b) for b in idx_bounds)):
            src_idx = _eval_remap(remapping_fn, dst_idx, phys_rank, idx_bounds)
            dst[dst_idx] = src_phys[src_idx]
        return SimLazyTensor(layout, SimTensor(layout, dst), is_view=True)

    grid = list(layout.grid_shape)
    block = list(layout.block_shape)
    tile = 32

    # Reshape the (element-space) physical buffer into an explicit tile grid:
    # each logical axis a splits into (grid[a], block[a], tile), then reorder to
    # [grid dims..., block dims..., tile dims...] so the leading 2N axes are the
    # affine index space the lambda addresses.
    split_shape = []
    for a in range(n):
        split_shape += [grid[a], block[a], tile]
    grid_view = src_phys.reshape(split_shape)
    grid_axes = [3 * a for a in range(n)]
    block_axes = [3 * a + 1 for a in range(n)]
    tile_axes = [3 * a + 2 for a in range(n)]
    perm = grid_axes + block_axes + tile_axes
    tg = grid_view.permute(*perm).contiguous()  # [grid..., block..., tile...]

    idx_bounds = grid + block  # extent of each of the 2N affine index dims
    dst = torch.empty_like(tg)
    for dst_idx in itertools.product(*(range(b) for b in idx_bounds)):
        src_idx = _eval_remap(remapping_fn, dst_idx, phys_rank, idx_bounds)
        dst[dst_idx] = tg[src_idx]

    # Undo the reshape/permute to return to element space.
    inv_perm = [0] * len(perm)
    for new_pos, old_axis in enumerate(perm):
        inv_perm[old_axis] = new_pos
    dst_phys = dst.permute(*inv_perm).contiguous().reshape(src_phys.shape)
    return SimLazyTensor(layout, SimTensor(layout, dst_phys), is_view=True)


def reshape(lt, *shape) -> SimLazyTensor:
    if not isinstance(lt, SimLazyTensor):
        raise TypeError(f"reshape expected a LazyTensor, got {type(lt).__name__}")
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        new_shape = tuple(shape[0])
    else:
        new_shape = tuple(shape)

    src_numel = math.prod(lt.layout.logical_shape)
    neg = [i for i, d in enumerate(new_shape) if d == -1]
    if len(neg) > 1:
        raise ValueError(
            f"reshape: only one dimension may be inferred (-1), got {new_shape}"
        )
    if neg:
        known = math.prod(d for d in new_shape if d != -1)
        if known == 0 or src_numel % known != 0:
            raise ValueError(
                f"reshape: cannot infer -1 dimension from {new_shape} (src numel {src_numel})"
            )
        new_shape = tuple(src_numel // known if d == -1 else d for d in new_shape)

    if math.prod(new_shape) != src_numel:
        raise ValueError(
            f"reshape: total element count must match: src {tuple(lt.layout.logical_shape)} "
            f"({src_numel}) != dst {new_shape} ({math.prod(new_shape)})"
        )

    rank = len(new_shape)
    dst_layout = lt.layout.replace(
        shape=new_shape, block_shape=[1] * rank, grid_shape=[1] * rank
    )
    host = lt._sim_logical().reshape(new_shape)
    return to_layout(host, dst_layout)


# --- materialisation --------------------------------------------------------


def to_host(*lts):
    if not lts:
        raise ValueError("to_host requires at least one LazyTensor")
    outs = []
    for i, lt in enumerate(lts):
        if lt.is_view:
            raise ValueError(
                f"to_host: argument {i} is a view (created via view/view_layout). "
                "Views are metadata reinterpretations and cannot be materialised "
                "directly; convert first, e.g. to_layout(v, v.layout)."
            )
        outs.append(lt._sim_logical())
    for lt, t in zip(lts, outs):
        lt.materialized = t
    _generation[0] += 1
    return tuple(outs)


# --- kernel dispatch --------------------------------------------------------


class SimCompiledKernel:
    """Runs a `@d2m.kernel` body as native Python via a globals rebind."""

    def __init__(self, fn, builtins):
        self.fn = fn
        # Shadow the kernel-body vocabulary (core_index, remote_load, sigmoid,
        # ...) into a copy of the function's own globals, then re-home the code
        # object onto it so `for`/`if`/`+=`/method-calls run natively.
        sim_globals = dict(fn.__globals__)
        sim_globals.update(builtins)
        self._sim_fn = types.FunctionType(
            fn.__code__, sim_globals, fn.__name__, fn.__defaults__, fn.__closure__
        )
        self.__name__ = getattr(fn, "__name__", "kernel")

    def __call__(
        self,
        *args,
        grid,
        num_outs: int = 1,
        block_factors=None,
        indexing_maps=None,
        iterator_types=None,
        kernel_io_in_dram=None,
    ):
        lazy_args, scalar_args, saw_scalar = [], [], False
        for i, a in enumerate(args):
            if isinstance(a, SimLazyTensor):
                if saw_scalar:
                    raise TypeError(
                        f"argument {i} is a LazyTensor after a scalar; tensor args "
                        "must precede scalars"
                    )
                lazy_args.append(a)
            elif isinstance(a, int) and not isinstance(a, bool):
                saw_scalar = True
                scalar_args.append(a)
            else:
                raise TypeError(
                    f"argument {i} has unsupported type {type(a).__name__}; kernel "
                    "args must be LazyTensor or int"
                )
        if num_outs < 1:
            raise ValueError(f"num_outs must be >= 1 (got {num_outs})")
        # block_factors / indexing_maps / iterator_types / kernel_io_in_dram do
        # not affect sim numerics; accepted and ignored.
        run_kernel(self._sim_fn, lazy_args, scalar_args, grid)
