# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""`d2m.topk`, the host-level chain: it plans a split across the worker grid and
emits transpose -> leaf sort -> (compact -> merge)* -> extract itself, so these
tests only supply shapes and a torch golden. The in-kernel `topk` builtin (one
core's shard, the piece the chain is built from) is exercised directly by the
rejection tests at the bottom."""

import functools

import pytest
import torch
import d2m_jit as d2m
from d2m_jit.api import (
    _TILE_WIDTH,
    _TOPK_L1_FAILURE_MARKERS,
    _TOPK_MAX_K,
    _TOPK_MIN_REDUCTION_TILES,
    _topk_merge_schedule,
)

pytestmark = pytest.mark.device_only(
    reason="asserts the device chain's split and the in-kernel topk's errors, "
    "neither of which the sim's torch.topk backing has"
)


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


# --- checks ------------------------------------------------------------------


def _layout(shape, dtype, block_shape, grid):
    return d2m.Layout(
        shape=tuple(shape),
        dtype=dtype,
        block_shape=list(block_shape),
        grid_shape=list(grid),
    )


def _check(shape, k, dim):
    """Plan + run through `d2m.topk`, then compare against a torch golden."""
    dim = dim % 2
    tensor = _planted_input(shape, k, dim)
    try:
        vals, idx = d2m.topk(tensor, k, dim).to_host()
    except Exception as exc:
        # No legal split on this grid, and a shard that overflows L1, are both
        # shape properties rather than regressions. `to_host` reports the
        # overflow as a D2mJitError, so neither can be caught by type.
        skippable = ("no legal split", *_TOPK_L1_FAILURE_MARKERS)
        if not any(marker in str(exc) for marker in skippable):
            raise
        pytest.skip(f"{shape} k={k} dim={dim}: {exc}")
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
    vals, idx = d2m.topk(tensor, 16, 1).to_host()

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
    """The chain's round splitting on its own, against a fixed grid so it stays
    readable. Reaches into `api` because `d2m.topk` plans internally."""
    assert _topk_merge_schedule(bands, (8, 8), cap) == want


@functools.lru_cache(maxsize=None)
def _sort_kernel(k, dim):
    """A bare leaf sort, for the rejection tests below: they check the errors the
    in-kernel `topk` builtin raises, which `d2m.topk` never reaches."""

    @d2m.kernel
    def kern(in_t, out_vals, out_idx):
        m = core_index(0)
        n = core_index(1)
        vals, idx = topk(remote_load(in_t, [m, n]), k, dim)
        remote_store(out_vals, [m, n], vals)
        remote_store(out_idx, [m, n], idx)

    return kern


def _reject(kernel, shape=(32, 128), dim=1):
    """Compile `kernel` and return the D2mJitError message it raises."""
    block = [1, 1]
    block[dim] = shape[dim] // _TILE_WIDTH
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
    msg = _reject(_sort_kernel(_TOPK_MAX_K + 1, 1))
    assert f"k must be in [1, {_TOPK_MAX_K}]" in msg, msg
    assert "test_topk.py" in msg, msg


def test_topk_single_tile_reduction_rejected():
    """One tile has nothing to merge against, so the pairwise sort cannot run."""
    msg = _reject(_sort_kernel(16, 1), shape=(_TILE_WIDTH, _TILE_WIDTH))
    assert f"at least {_TOPK_MIN_REDUCTION_TILES} tiles" in msg, msg
