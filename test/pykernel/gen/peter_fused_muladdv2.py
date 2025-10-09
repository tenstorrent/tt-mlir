# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.kernel_ast import *
from utils import assert_pcc
import torch


@pykernel_gen(
    grid=(2, 2),
    kernel_source_mode="store",
    verbose=False,
    granularity=4, # compute granularity. could be passed by user, or left for auto-tuning
)
def muladd(a_in, b_in, c_in, out, grid=None):
    assert grid is not None
    
    # Assuming lightweight op input validation should be here
    assert a_in.shape == b_in.shape == c_in.shape == out.shape

    row_tiles = a_in.shape[0] // TILE_SIZE
    col_tiles = a_in.shape[1] // TILE_SIZE
    
    # Assuming parallelized by rows
    rows_per_core = math.ceil(row_tiles / (grid[0] * grid[1]))
    
    """
    This variant of muladd experiments with the idea of giving Streams a different
    index type. You can imagine that indexing in TILEs is the best supported, most
    performant method. Setting an index_type prevents the user from invoking unaligned,
    non-trivial reads.
    
    The compiler could always figure out when you're indexing tiles, but it can help
    the user to enforce performant indexing.
    
    Perhaps the index type should be specified in the `dma` call rather than the Stream constructor,
    but that's a detail.
    """
    a_accessor = Stream(a_in, index_type=IndexType.TILE)
    b_accessor = Stream(b_in, index_type=IndexType.TILE)
    c_accessor = Stream(c_in, index_type=IndexType.TILE)
    out_accessor = Stream(out, index_type=IndexType.TILE)
    
    # Define CBs that interface between datamovement and compute
    # Intermediate CBs internal to compute can be automatically determined

    # PETER: CBs are inferred--see below
    
    # a_in_cb = CircularBuffer(a_accessor, shape=(granularity,), buffer_factor=2)
    # b_in_cb = CircularBuffer(b_accessor, shape=(granularity,), buffer_factor=2)
    # c_in_cb = CircularBuffer(c_accessor, shape=(granularity,), buffer_factor=2)
    # out_cb = CircularBuffer(out_accessor, shape=(granularity,), buffer_factor=2)

    core_num = core_index() # core number in 2d grid
    start_row_tile = core_num * rows_per_core
    end_row_tile = min(start_row_tile + rows_per_core, row_tiles)
    
    for rt in range(start_row_tile, end_row_tile):
        for ct_block in range(col_tiles // granularity):
            row_slice = slice(rt, rt+1)
            col_slice = slice(ct_block*granularity, (ct_block+1)*granularity)

            # PETER: a_in_cb and its shape is inferred as (granularity,1) based on row_slice and col_slice

            a_block = load(a_accessor[row_slice, col_slice])

            # PETER: load codegened in dm0
            # a_in_cb.reserve()
            # tx = dma(a_accessor[row_slice, col_slice], a_in_cb)
            # tx.wait()
            # a_in_cb.push()

            # PETER: load codegened in compute
            # a_block = a_in_cb.wait().pop()

            # PETER: b_in_cb and its shape is inferred as (granularity,1) based on row_slice and col_slice

            b_block = load(b_accessor[row_slice, col_slice])

            # PETER: load codegened in dm0
            # b_in_cb.reserve()
            # tx = dma(b_accessor[row_slice, col_slice], b_in_cb)
            # tx.wait()
            # b_in_cb.push()

            # PETER: load codegened in compute
            # b_block = b_in_cb.wait().pop()

            # PETER: c_in_cb and its shape is inferred as (granularity,1) based on row_slice and col_slice

            c_block = load(c_accessor[row_slice, col_slice])

            # PETER: load codegened in dm0
            # c_in_cb.reserve()
            # tx = dma(c_accessor[row_slice, col_slice], c_in_cb)
            # tx.wait()
            # c_in_cb.push()

            # PETER: load codegened in compute
            # c_block = c_in_cb.wait().pop()

            out_block = a_block * b_block + c_block

            # PETER: out_cb and its shape is inferred as (granularity,1) based on row_slice and col_slice

            store(out_block, out_accessor[row_slice, col_slice])

            # PETER: store codegened in compute
            # out_cb.push(out_block)

            # PETER: store codegened in dm1
            # out_cb.wait()
            # tx = dma(out_cb, out_accessor[row_slice, col_slice])
            # tx.wait()
            # out_cb.pop()

a_in = torch.randn(128, 128)
b_in = torch.randn(128, 128)
c_in = torch.randn(128, 128)
out = torch.zeros(128, 128)
muladd(a_in, b_in, c_in, out)

golden = a_in * b_in + c_in
assert_pcc(golden, out)
