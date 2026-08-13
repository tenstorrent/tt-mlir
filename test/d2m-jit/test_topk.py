# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Block-level `topk` via `d2m.topk_block`, driven as an explicit kernel chain
rather than through ttir.topk. Banded shapes add a `topk_extract(transpose=
False)` + `topk_merge` round between the leaf sort and the final extract."""

import functools

import pytest
import torch
import d2m_jit as d2m

pytestmark = pytest.mark.device_only(
    reason="_src/sim/ops.py does not register topk, so the simulator cannot "
    "dispatch the op at all -- neither the correctness nor the rejection tests "
    "have anything to run against"
)

_TILE = 32
# Reduction tiles per core to aim for when banding. Not an L1 bound -- `_check`
# catches a real overflow; this just keeps large shapes off the single-core
# split so the merge path stays exercised.
_TILES_PER_CORE = 48


# --- kernels -----------------------------------------------------------------
# k and dim are attributes on d2m.topk_block, not runtime arguments, so each
# kernel is specialised per (k, dim).


@functools.lru_cache(maxsize=None)
def _transpose_kernel():
    @d2m.kernel
    def kern(in_t, out_t):
        m = core_index(0)
        n = core_index(1)
        remote_store(out_t, [m, n], tile_transpose(remote_load(in_t, [m, n])))

    return kern


@functools.lru_cache(maxsize=None)
def _sort_kernel(k, dim):
    @d2m.kernel
    def kern(in_t, out_vals, out_idx):
        m = core_index(0)
        n = core_index(1)
        vals, idx = topk(remote_load(in_t, [m, n]), k, dim)
        remote_store(out_vals, [m, n], vals)
        remote_store(out_idx, [m, n], idx)

    return kern


@functools.lru_cache(maxsize=None)
def _merge_kernel(k, dim):
    @d2m.kernel
    def kern(vals_t, idx_t, out_vals, out_idx):
        m = core_index(0)
        n = core_index(1)
        vals, idx = topk_merge(
            remote_load(vals_t, [m, n]), remote_load(idx_t, [m, n]), k, dim
        )
        remote_store(out_vals, [m, n], vals)
        remote_store(out_idx, [m, n], idx)

    return kern


@functools.lru_cache(maxsize=None)
def _compact_kernel(k, dim):
    @d2m.kernel
    def kern(in_t, out_t):
        m = core_index(0)
        n = core_index(1)
        remote_store(
            out_t,
            [m, n],
            topk_extract(remote_load(in_t, [m, n]), k, dim, transpose=False),
        )

    return kern


@functools.lru_cache(maxsize=None)
def _extract_kernel(k, dim):
    @d2m.kernel
    def kern(in_t, out_t):
        m = core_index(0)
        n = core_index(1)
        remote_store(out_t, [m, n], topk_extract(remote_load(in_t, [m, n]), k, dim))

    return kern


# --- planning ----------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _worker_grid():
    """The device's worker grid, which every split is ultimately folded onto."""
    from d2m_jit._src.builder import _device_worker_grid

    return _device_worker_grid()


def _fits_grid(cores, worker_grid):
    """True when `cores` factors into a rectangle the worker grid holds."""
    from d2m_jit._src.builder import _legal_physical_grid

    return _legal_physical_grid(cores, worker_grid) is not None


def _merge_schedule(bands, merge_cap, worker_grid):
    """Group count surviving each merge round, outermost first, ending at 1.
    None when no round satisfies `merge_cap` and the grid."""
    schedule = []
    while bands > 1:
        groups = next(
            (
                g
                for g in range(1, bands)
                if bands % g == 0
                and bands // g <= merge_cap
                and _fits_grid(g, worker_grid)
            ),
            None,
        )
        if groups is None:
            return None
        schedule.append(groups)
        bands = groups
    return schedule


def _split_order(cores, band_limit, nt_limit, nt_tiles, red_tiles):
    """Candidate (bands, nt_shards) splits, most-spread first: bands enough to
    hit `_TILES_PER_CORE`, then coarser, then data-parallel, then both, then
    one core."""
    wanted = min(cores, max(1, -(-red_tiles // _TILES_PER_CORE)))
    for bands in range(wanted, 1, -1):
        yield bands, 1
    for nt_shards in range(min(nt_tiles, cores), 1, -1):
        yield 1, nt_shards
    for bands in range(band_limit, 1, -1):
        for nt_shards in range(2, nt_limit + 1):
            yield bands, nt_shards
    yield 1, 1


def _plan(shape, k, dim, merge_cap=4):
    """Bands and non-target shards for `shape`, or None when no split is legal."""
    worker_grid = _worker_grid()
    cores = worker_grid[0] * worker_grid[1]
    band_limit = worker_grid[1] if dim == 1 else min(worker_grid)
    nt_limit = worker_grid[0] if dim == 1 else worker_grid[1]

    full_red_tiles = -(-shape[dim] // _TILE)
    # topk_block merges reduction tiles pairwise, so a band is never one tile.
    local_red_tiles = max(2, full_red_tiles)
    nt_tiles = -(-shape[1 - dim] // _TILE)

    def candidate(bands, nt_shards):
        band_tiles = local_red_tiles if bands == 1 else -(-full_red_tiles // bands)
        nt_per_core = -(-nt_tiles // nt_shards)
        if bands > 1 and band_tiles < 2:
            return None
        placeable = (
            bands <= band_limit and nt_shards <= nt_limit
            if bands > 1 and nt_shards > 1
            else _fits_grid(bands * nt_shards, worker_grid)
        )
        if not placeable:
            return None
        if bands > 1 and _merge_schedule(bands, merge_cap, worker_grid) is None:
            return None
        return bands, band_tiles, nt_shards, nt_per_core

    # First legal split wins; `_check` skips it if L1 overflows.
    for bands, nt_shards in _split_order(
        cores, band_limit, nt_limit, nt_tiles, full_red_tiles
    ):
        entry = candidate(bands, nt_shards)
        if entry is not None:
            return entry
    return None


# --- input / reference -------------------------------------------------------


def _planted_input(shape, k, dim):
    """Uniform noise in [0, 1) with `k` winners planted per reduction slice.
    Winners are spaced a whole unit apart so precision loss cannot create ties."""
    tensor = torch.rand(shape)
    positions = torch.rand(shape).argsort(dim=dim).narrow(dim, 0, k)
    winners = torch.arange(k, dtype=torch.float32) + 10.0
    broadcast = [1] * len(shape)
    broadcast[dim] = k
    tensor.scatter_(dim, positions, winners.reshape(broadcast).expand_as(positions))
    return tensor


def _pad_for_tiles(tensor, dim, red_tiles, nt_tiles):
    """Pad to whole tiles: the reduction with -inf so padding can never win,
    the non-target with zeros since those slices are sliced back off."""
    padded = list(tensor.shape)
    padded[dim] = red_tiles * _TILE
    padded[1 - dim] = nt_tiles * _TILE
    if tuple(padded) == tuple(tensor.shape):
        return tensor
    out = torch.zeros(padded, dtype=tensor.dtype)
    out.narrow(dim, tensor.shape[dim], padded[dim] - tensor.shape[dim]).fill_(
        float("-inf")
    )
    out[tuple(slice(0, s) for s in tensor.shape)] = tensor
    return out


def _assert_topk(vals, idx, tensor, k, dim, value_atol=0.002):
    """Check which elements were selected (exact), then how precisely their
    values came back (tolerance). Both sort along `dim` first since order is
    not under test."""
    expected = torch.topk(tensor, k, dim=dim)

    assert idx.dtype == torch.int32, idx.dtype
    assert idx.min().item() >= 0, f"negative index {idx.min().item()}"
    reduction_size = tensor.shape[dim]
    assert (
        idx.max().item() < reduction_size
    ), f"index {idx.max().item()} outside [0, {reduction_size})"

    want = expected.values.sort(dim=dim).values
    selected = tensor.gather(dim, idx.to(torch.int64)).sort(dim=dim).values
    torch.testing.assert_close(
        selected,
        want,
        rtol=0,
        atol=0,
        msg=lambda default: f"topk selected the wrong elements: {default}",
    )

    diff = (vals.to(torch.float32).sort(dim=dim).values - want).abs().max().item()
    assert diff < value_atol, (
        f"topk selected the right elements but the returned values lost "
        f"precision: max diff {diff} (tolerance {value_atol})"
    )


# --- chain -------------------------------------------------------------------


def _layout(shape, dtype, block_shape, grid):
    return d2m.Layout(
        shape=tuple(shape),
        dtype=dtype,
        block_shape=list(block_shape),
        grid_shape=list(grid),
    )


def _run_topk_chain(tensor, k, dim, plan, merge_cap=4):
    """transpose -> leaf sort -> (compact -> merge)* -> extract, one kernel per
    `d2m.generic`. Stays in the leaf's transposed orientation until the final
    extract; transposing between rounds would desynchronise values from indices."""
    bands, band_tiles, nt_shards, nt_per_core = plan
    logical = tuple(tensor.shape)
    red_tiles, nt_tiles = bands * band_tiles, nt_shards * nt_per_core
    padded = _pad_for_tiles(tensor, dim, red_tiles, nt_tiles)

    def axes(reduction, non_target):
        return [non_target, reduction] if dim == 1 else [reduction, non_target]

    grid = tuple(axes(bands, nt_shards))
    compact_tiles = -(-k // _TILE)

    def shaped(reduction_tiles):
        return tuple(x * _TILE for x in axes(reduction_tiles, nt_tiles))

    def buf(reduction_tiles, block_red, dtype, g):
        return d2m.empty(
            _layout(
                shaped(reduction_tiles), dtype, axes(block_red, nt_per_core), tuple(g)
            )
        )

    src = d2m.to_layout(
        padded,
        _layout(shaped(red_tiles), d2m.float32, axes(band_tiles, nt_per_core), grid),
    )
    if dim == 1:
        turned = buf(red_tiles, band_tiles, d2m.float32, grid)
        _transpose_kernel()(src, turned, grid=grid)
        src = turned

    vals = buf(red_tiles, band_tiles, d2m.float32, grid)
    idxs = buf(red_tiles, band_tiles, d2m.int32, grid)
    _sort_kernel(k, dim)(src, vals, idxs, grid=grid, num_outs=2)

    if bands > 1:
        groups = bands
        live = groups * compact_tiles
        part_vals = buf(live, compact_tiles, d2m.float32, grid)
        part_idx = buf(live, compact_tiles, d2m.int32, grid)
        _compact_kernel(k, dim)(vals, part_vals, grid=grid)
        _compact_kernel(k, dim)(idxs, part_idx, grid=grid)
        vals, idxs = part_vals, part_idx

        for survivors in _merge_schedule(bands, merge_cap, _worker_grid()):
            fan_in = groups // survivors
            round_grid = tuple(axes(survivors, nt_shards))
            block_red = fan_in * compact_tiles
            gathered = axes(block_red, nt_per_core)
            vals = d2m.to_layout(
                vals, _layout(shaped(live), d2m.float32, gathered, round_grid)
            )
            idxs = d2m.to_layout(
                idxs, _layout(shaped(live), d2m.int32, gathered, round_grid)
            )
            merged_vals = buf(live, block_red, d2m.float32, round_grid)
            merged_idx = buf(live, block_red, d2m.int32, round_grid)
            _merge_kernel(k, dim)(
                vals, idxs, merged_vals, merged_idx, grid=round_grid, num_outs=2
            )

            groups, live = survivors, survivors * compact_tiles
            vals = buf(live, compact_tiles, d2m.float32, round_grid)
            idxs = buf(live, compact_tiles, d2m.int32, round_grid)
            _compact_kernel(k, dim)(merged_vals, vals, grid=round_grid)
            _compact_kernel(k, dim)(merged_idx, idxs, grid=round_grid)
        grid = tuple(axes(1, nt_shards))

    out_grid = grid
    out_logical = list(shaped(compact_tiles))
    out_logical[dim] = k
    out_vals = d2m.empty(
        _layout(out_logical, d2m.float32, axes(compact_tiles, nt_per_core), out_grid)
    )
    out_idx = d2m.empty(
        _layout(out_logical, d2m.int32, axes(compact_tiles, nt_per_core), out_grid)
    )
    _extract_kernel(k, dim)(vals, out_vals, grid=out_grid)
    _extract_kernel(k, dim)(idxs, out_idx, grid=out_grid)

    host_vals, host_idx = d2m.to_host(out_vals, out_idx)
    keep = [slice(None), slice(None)]
    keep[1 - dim] = slice(0, logical[1 - dim])
    return host_vals[tuple(keep)], host_idx[tuple(keep)]


def _check(shape, k, dim):
    dim = dim % 2
    plan = _plan(shape, k, dim)
    if plan is None:
        pytest.skip(
            f"{shape} k={k} dim={dim} has no legal split on a "
            f"{list(_worker_grid())} grid: no band count leaves a band of two "
            "tiles with a merge tree the grid can hold, and no non-target "
            "slice fits either"
        )
    tensor = _planted_input(shape, k, dim)
    try:
        vals, idx = _run_topk_chain(tensor, k, dim, plan)
    except Exception as exc:
        # An L1 overflow is a shape property, not a regression.
        if "exceeds memory capacity" not in str(exc):
            raise
        pytest.skip(f"{shape} k={k} dim={dim} split {plan} does not fit L1: {exc}")
    assert tuple(vals.shape) == tuple(s if i != dim else k for i, s in enumerate(shape))
    _assert_topk(vals, idx, tensor, k, dim)


# --- shapes, ported from test/python/golden/d2m/test_topk.py ------------------

SINGLE_CORE_TOPK_SHAPES = [
    pytest.param((32, 256), 64, -1, id="32x256_k64_dim1"),
    pytest.param((256, 32), 64, 0, id="256x32_k64_dim0"),
    pytest.param((32, 1376), 64, -1, id="32x1376_k64_dim1"),
    pytest.param((1376, 32), 64, 0, id="1376x32_k64_dim0"),
    pytest.param((32, 96), 16, -1, id="32x96_k16_dim1"),
    pytest.param((1208, 32), 16, 0, id="1208x32_k16_dim0"),
    pytest.param((96, 446), 32, -1, id="96x446_k32_dim1"),
    pytest.param((383, 96), 63, 0, id="383x96_k63_dim0"),
]

MULTI_CORE_TOPK_SHAPES = [
    pytest.param((32, 5504), 16, -1, id="32x5504_k16_dim1"),
    pytest.param((32, 96256), 16, -1, id="32x96256_k16_dim1"),
    pytest.param((35, 7639), 16, -1, id="35x7639_k16_dim1"),
    pytest.param((8192, 32), 16, 0, id="8192x32_k16_dim0"),
    pytest.param((96256, 32), 16, 0, id="96256x32_k16_dim0"),
    pytest.param((7639, 35), 16, 0, id="7639x35_k16_dim0"),
    pytest.param((32, 8192), 48, -1, id="32x8192_k48_dim1"),
    pytest.param((32, 96256), 64, -1, id="32x96256_k64_dim1"),
    pytest.param((8192, 32), 48, 0, id="8192x32_k48_dim0"),
    pytest.param((96256, 32), 64, 0, id="96256x32_k64_dim0"),
    pytest.param((32, 8192), 16, 0, id="32x8192_k16_dim0"),
    pytest.param((8192, 32), 16, -1, id="8192x32_k16_dim1"),
]


# --- tests -------------------------------------------------------------------


def test_topk_compiles_and_runs():
    tensor = _planted_input((32, 128), 16, 1)
    vals, idx = _run_topk_chain(tensor, 16, 1, _plan((32, 128), 16, 1))

    assert tuple(vals.shape) == (32, 16)
    assert tuple(idx.shape) == (32, 16)
    assert vals.dtype == torch.float32
    assert idx.dtype == torch.int32


@pytest.mark.parametrize("shape,k,dim", SINGLE_CORE_TOPK_SHAPES)
def test_topk_single_core(shape, k, dim):
    _check(shape, k, dim)


@pytest.mark.parametrize("shape,k,dim", MULTI_CORE_TOPK_SHAPES)
def test_topk_multi_core(shape, k, dim):
    _check(shape, k, dim)


@pytest.mark.parametrize(
    "bands,cap,want",
    [(1, 4, []), (4, 4, [1]), (8, 4, [2, 1]), (8, 2, [4, 2, 1]), (12, 3, [4, 2, 1])],
)
def test_topk_merge_schedule(bands, cap, want):
    """Round splitting on its own, against a fixed grid so it stays readable."""
    assert _merge_schedule(bands, cap, (8, 8)) == want


def _reject(kernel, shape=(32, 128), dim=1):
    """Compile `kernel` and return the D2mJitError message it raises."""
    block = [1, 1]
    block[dim] = shape[dim] // _TILE
    layout = _layout(shape, d2m.float32, block, (1, 1))
    with pytest.raises(d2m.D2mJitError) as exc_info:
        kernel(
            d2m.to_layout(torch.zeros(*shape, dtype=torch.float32), layout),
            d2m.empty(layout),
            d2m.empty(_layout(shape, d2m.int32, block, (1, 1))),
            grid=(1, 1),
            num_outs=2,
        )
    return str(exc_info.value)


def test_topk_k_above_max_rejected():
    msg = _reject(_sort_kernel(128, 1))
    assert "k must be in [1, 64]" in msg, msg
    assert "test_topk.py" in msg, msg


def test_topk_single_tile_reduction_rejected():
    """One tile has nothing to merge against, so the pairwise sort cannot run."""
    msg = _reject(_sort_kernel(16, 1), shape=(32, 32))
    assert "at least 2 tiles" in msg, msg
