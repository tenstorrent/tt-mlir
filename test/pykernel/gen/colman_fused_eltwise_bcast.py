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

    a_accessor = TensorAccessor(a_in, index_type=IndexType.TILE)
    b_accessor = TensorAccessor(b_in, index_type=IndexType.TILE)
    c_accessor = TensorAccessor(c_in, index_type=IndexType.TILE)
    out_accessor = TensorAccessor(out, index_type=IndexType.TILE)
    
    # NOTE: (Kostas) I don’t understand why a CircularBuffer needs to be associated with a tensor accessor.
    #                Perhaps we need to know its specific type? Or to prevent mixups of tensors on the same cb?
    a_in_cb = CircularBuffer(a_accessor, shape=(granularity,1), buffer_factor=2)
    b_in_cb = CircularBuffer(b_accessor, shape=(granularity,1), buffer_factor=2)
    # NOTE: should it be declared somewhere that c_in_cb contains a row padded to tile,
    # so that compute should choose bcast eltwise ops?
    c_in_cb = CircularBuffer(c_accessor, shape=(1,1), buffer_factor=2)
    out_cb = CircularBuffer(out_accessor, shape=(granularity,1), buffer_factor=2)

    @compute()
    async def compute():
        core_num = core_index() # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
        
        for ct in range(start_col_tile, end_col_tile):
            # Reuse C across rows of A, B
            # From a sim perspective, the following returns a RingView object as defined in:
            # https://github.com/tenstorrent/tt-lang/blob/main/python/sim/cbsim/ringview.py.
            # TODO: Perhaps consider making RingView pointers that come from wait()/reserve() read/write only respectively?
            c_block = c_in_cb.wait() 
            for rt_block in range(row_tiles // granularity):
                # again, these return RingView pointers:
                a_block = a_in_cb.wait() # blocking 
                b_block = b_in_cb.wait() # blocking
                # NOTE: Please consider making non-approx the default for eltwise unary, but leave the option for the user to specify approx=True
                out_block = out_cb.reserve() # blocking
                # RingView operations, these also need to be defined
                out_block = a_block * b_block + tt.tanh(c_block, approx=False)
                # finalize push, this advances the cb pointers, the writing happened at the line above
                out_cb.push()
                # finalize pop, this advances the cb pointers, essentially freeing the memory
                # After poping, the corresponding RingView(a_block) points to stale data.
                # Should probably make it an error to access it at that point, at least in sim.
                a_in_cb.pop()
                # ditto
                b_in_cb.pop()
            c_in_cb.pop() # After this pop, c_block points to stale data.


    @datamovement()
    async def dm0():
        core_num = core_index() # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
        
        c_row_slice = slice(0, 1)
        for ct in range(start_col_tile, end_col_tile):
            # Reuse C across rows of A, B
            c_col_slice = slice(ct, ct+1)
            c_block = c_in_cb.reserve() # retrieve a RingView to the cb
	    # write to the cb through a DMA. This is non-blocking
            tx = dma(c_accessor[c_row_slice, c_col_slice], c_block)
            # Wait for the dma to conclude before continuing
            tx.wait()
            # Finalize the write on the cb (advances cb pointers)
            c_in_cb.push()
            for rt_block in range(row_tiles // granularity):
                """
                Since the TensorAccessor indexes by tile, slicing is cleaner
                """
                row_slice = slice(rt_block*granularity, (rt_block+1)*granularity)
                col_slice = slice(ct, ct+1)
                # Write the cbs just as above
                a_block = a_in_cb.reserve()
                tx = dma(a_accessor[row_slice, col_slice], a_block)
                tx.wait()
                a_in_cb.push()
                b_block = b_in_cb.reserve()
                tx = dma(b_accessor[row_slice, col_slice], b_block)
                tx.wait()
                b_in_cb.push()

    @datamovement()
    async def dm1():
        core_num = core_index() # core number in 2d grid
        start_col_tile = core_num * cols_per_core
        end_col_tile = min(start_col_tile + cols_per_core, col_tiles)
        
        c_row_slice = slice(0, 1)
        for ct in range(start_col_tile, end_col_tile):
            # Reuse C across rows of A, B
            c_col_slice = slice(ct, ct+1)
            c_block = c_in_cb.reserve()
            tx = dma(c_accessor[c_row_slice, c_col_slice], c_block)
            tx.wait()
            c_in_cb.push()
            for rt_block in range(row_tiles // granularity):
                """
                Since the TensorAccessor indexes by tile, slicing is cleaner
                """
                row_slice = slice(rt_block*granularity, (rt_block+1)*granularity)
                col_slice = slice(ct, ct+1)
                out_block = out_cb.wait()
                tx = dma(out_block, out_accessor[row_slice, col_slice])
                tx.wait()
                out_block.pop()


    return Program(compute, dm0, dm1)(a_in, b_in, c_in, out)

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