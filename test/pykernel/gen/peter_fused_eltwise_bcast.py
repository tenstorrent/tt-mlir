# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.kernel_ast import *
from utils import assert_pcc
import torch


@pykernel_gen(
    grid='auto', # NOTE: allow compiler to choose grid
    granularity=4, # compute granularity. could be passed by user, or left for auto-tuning
)
def eltwise_bcast(a_in, b_in, c_in, out, grid=None):
    assert grid is not None
    
    # Assuming lightweight op input validation should be here
    assert a_in.shape == b_in.shape == out.shape
    assert c_in.shape == (1, a_in.shape[1]) # checking logical shape
    assert all(is_tiled(tensor) for tensor in [a_in, b_in, c_in, out])
    assert a_in.shape[0] % granularity == 0

    row_tiles = a_in.shape[0] // TILE_SIZE
    col_tiles = a_in.shape[1] // TILE_SIZE
    
    # Parallelizing by columns here to get reuse on C
    cols_per_core = math.ceil(col_tiles / (grid[0] * grid[1]))

    a_accessor = Stream(a_in, index_type=IndexType.TILE)
    b_accessor = Stream(b_in, index_type=IndexType.TILE)
    c_accessor = Stream(c_in, index_type=IndexType.TILE)
    out_accessor = Stream(out, index_type=IndexType.TILE)

    # PETER: CBs are inferred--see below
    
    # a_in_cb = CircularBuffer(a_accessor, shape=(granularity,1), buffer_factor=2)
    # b_in_cb = CircularBuffer(b_accessor, shape=(granularity,1), buffer_factor=2)
    # NOTE: should it be declared somewhere that c_in_cb contains a row padded to tile,
    # so that compute should choose bcast eltwise ops?
    # c_in_cb = CircularBuffer(c_accessor, shape=(1,1), buffer_factor=2)
    # out_cb = CircularBuffer(out_accessor, shape=(granularity,1), buffer_factor=2)

    core_num = core_index() # core number in 2d grid
    start_col_tile = core_num * cols_per_core
    end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
    c_row_slice = slice(0, 1)
        
    for ct in range(start_col_tile, end_col_tile):
        c_col_slice = slice(ct, ct+1)

        # Reuse C across rows of A, B

        # PETER: b_in_cb and its shape is inferred as (granularity,1) based on c_row_slice and c_col_slice

        c_block = load(c_accessor[c_row_slice, c_col_slice])

        # PETER: load codegened in dm0
        # c_in_cb.reserve()
        # tx = dma(c_accessor[c_row_slice, c_col_slice], c_in_cb)
        # tx.wait()
        # c_in_cb.push()

        # PETER: load codegened in compute
        # c_block = c_in_cb.wait().pop()

        for rt_block in range(row_tiles // granularity):
            row_slice = slice(rt_block*granularity, (rt_block+1)*granularity)
            col_slice = slice(ct, ct+1)

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

            # NOTE: Please consider making non-approx the default for eltwise unary, but leave the option for the user to specify approx=True
            out_block = a_block * b_block + tt.tanh(c_block, approx=False)

            # PETER: out_cb and its shape is inferred as (granularity,1) based on row_slice and col_slice

            store(out_block, out_accessor[row_slice, col_slice])

            # PETER: store codegened in compute
            # out_cb.push(out_block)

            # PETER: store codegened in dm1
            # out_cb.wait()
            # tx = dma(out_cb, out_accessor[row_slice, col_slice])
            # tx.wait()
            # out_cb.pop()

"""
out = a * b + tanh(c)

where c is broadcasted on rows
"""

a_in = torch.randn(128, 128)
b_in = torch.randn(128, 128)
# Assume that c_in is a tilized (padded) row vector
c_in = torch.randn(1, 128)
out = torch.zeros(128, 128)
eltwise_bcast(a_in, b_in, c_in, out)

golden = a_in * b_in + torch.tanh(c_in)
assert_pcc(golden, out)
