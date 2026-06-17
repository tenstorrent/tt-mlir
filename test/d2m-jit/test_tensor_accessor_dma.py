# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for the TensorAccessor-based shard-level DMA lowering
(`use-tensor-accessor-dma`).

The runtime BufferDistributionSpec, the TensorAccessor, and the tensor strides
all operate in *page* units. For tiled buffers the element is itself a page
(one tile), but for row-major (scalar-element) buffers the page is a stick of
`(1, shardWidth)` (matching `createShardedBufferConfigForL1Memref`). The
accessor read/write loop in `D2MDMAViaTensorAccessorRewriter` must therefore
iterate in page units too: it previously looped over the *element* shard shape
with an *element* page size while the tensor stride was stick-based, so a
row-major shard was addressed with the wrong row stride and the tiles were
scrambled (e.g. through to_layout's tilize, which reads the row-major source).

`test_accessor_dma_copy_through_view` is the regression test: a pure copy
(remote_load -> remote_store, no compute) through such a layout.
"""

import pytest

import d2m_jit as d2m
from d2m_jit._src.config import config
from d2m_jit.api import core_index, remote_load, remote_store

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    from _ttmlir_runtime import runtime
except (ModuleNotFoundError, ImportError):
    runtime = None

from utils import arange_tile, assert_pcc


@pytest.fixture
def tensor_accessor_dma():
    """Enable the TensorAccessor DMA lowering for the duration of one test."""
    old = config.use_tensor_accessor_dma
    config.use_tensor_accessor_dma = True
    try:
        yield
    finally:
        config.use_tensor_accessor_dma = old


@d2m.kernel
def _copy(inp, out, m_blocks, n_blocks):
    cy = core_index(0)
    cx = core_index(1)
    m_off = cy * m_blocks
    n_off = cx * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(inp, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], x)


def _run_copy(grid_shape, compute_grid):
    src = arange_tile(256, 256, dtype=torch.float)
    m_blocks = (256 // 32) // compute_grid[0]
    n_blocks = (256 // 32) // compute_grid[1]
    L = d2m.Layout(
        shape=src.shape, dtype=src.dtype, block_shape=[1, 1], grid_shape=grid_shape
    )
    in_d = d2m.to_layout(src, L)
    out_d = d2m.empty(L)
    _copy(in_d, out_d, m_blocks, n_blocks, grid=compute_grid)
    return src, out_d.to_host()


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
def test_fully_indexed_copy_through_view():
    """Control: the fully-indexed path copies correctly through the view."""
    src, out = _run_copy(grid_shape=[4, 4], compute_grid=(4, 4))
    assert_pcc(src, out)


@pytest.mark.skipif(
    torch is None or runtime is None, reason="requires torch + ttmlir runtime"
)
def test_accessor_dma_copy_through_view(tensor_accessor_dma):
    """The accessor path must apply the same views as the fully-indexed path.
    A grid coarser than the tile grid (block_shape [1,1], grid [4,4] over an
    8x8-tile tensor) induces a non-identity view; if it is dropped, the tiles
    are scrambled."""
    src, out = _run_copy(grid_shape=[4, 4], compute_grid=(4, 4))
    assert_pcc(src, out)
