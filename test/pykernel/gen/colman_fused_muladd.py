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
    
    # Does the user really need to create these streams or can they be inferred?
    a_accessor = Stream(a_in)
    b_accessor = Stream(b_in)
    c_accessor = Stream(c_in)
    out_accessor = Stream(out)
    
    # Define CBs that interface between datamovement and compute
    # Intermediate CBs internal to compute can be automatically determined
    a_in_cb = CircularBuffer(a_accessor, shape=(granularity,), buffer_factor=2)
    b_in_cb = CircularBuffer(b_accessor, shape=(granularity,), buffer_factor=2)
    c_in_cb = CircularBuffer(c_accessor, shape=(granularity,), buffer_factor=2)
    out_cb = CircularBuffer(out_accessor, shape=(granularity,), buffer_factor=2)

    @compute()
    async def compute():
        core_num = core_index() # core number in 2d grid
        start_row_tile = core_num * rows_per_core
        end_row_tile = min(start_row_tile + rows_per_core, row_tiles)
        
        for rt in range(start_row_tile, end_row_tile):
            for ct_block in range(col_tiles // granularity):
                # I'm not sure this is the right syntax. In compute kernel, pop means 'give me a handle to a block of data for use in compute'. In writer, pop just frees the block from the CB.
                a_block = a_in_cb.wait().pop()
                b_block = b_in_cb.wait().pop()
                c_block = c_in_cb.wait().pop()
                out_block = a_block * b_block + c_block
                out_cb.push(out_block)

    @datamovement()
    async def dm0():
        core_num = core_index() # core number in 2d grid
        start_row_tile = core_num * rows_per_core
        end_row_tile = min(start_row_tile + rows_per_core, row_tiles)
        
        for rt in range(start_row_tile, end_row_tile):
            for ct_block in range(col_tiles // granularity):
                row_slice = slice(rt*TILE_SIZE, (rt+1)*TILE_SIZE)
                col_slice = slice(ct_block*granularity*TILE_SIZE, (ct_block+1)*granularity*TILE_SIZE)
                a_in_cb.reserve()
                tx = dma(a_accessor[row_slice, col_slice], a_in_cb)
                tx.wait()
                a_in_cb.push()
                b_in_cb.reserve()
                tx = dma(b_accessor[row_slice, col_slice], b_in_cb)
                tx.wait()
                b_in_cb.push()
                c_in_cb.reserve()
                tx = dma(c_accessor[row_slice, col_slice], c_in_cb)
                tx.wait()
                c_in_cb.push()

    @datamovement()
    async def dm1():
        core_num = core_index() # core number in 2d grid
        start_row_tile = core_num * rows_per_core
        end_row_tile = min(start_row_tile + rows_per_core, row_tiles)
        
        for rt in range(start_row_tile, end_row_tile):
            for ct_block in range(col_tiles // granularity):
                row_slice = slice(rt*TILE_SIZE, (rt+1)*TILE_SIZE)
                col_slice = slice(ct_block*granularity*TILE_SIZE, (ct_block+1)*granularity*TILE_SIZE)
                out_cb.wait()
                tx = dma(out_cb, out_accessor[row_slice, col_slice])
                tx.wait()
                out_cb.pop()

    return Program(compute, dm0, dm1)(a_in, b_in, c_in, out)


a_in = torch.randn(128, 128)
b_in = torch.randn(128, 128)
c_in = torch.randn(128, 128)
out = torch.zeros(128, 128)
muladd(a_in, b_in, c_in, out)

golden = a_in * b_in + c_in
assert_pcc(golden, out)
