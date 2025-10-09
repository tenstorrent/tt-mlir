# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.kernel_ast import *
from utils import assert_pcc
import torch


@pykernel_gen(
    grid='auto', # NOTE: allow compiler to choose grid
)
def create_qkv_heads(q_in, kv_in, q_out, k_out, v_out, num_heads, grid=None):
    assert grid is not None

    # NOTE: not validated here, but dim -2 can be padded to tile
    B, S_query = q_in.padded_shape[:2]
    S_kv = kv_in.padded_shape[1]
    D = q_in.shape[2]
    DH = D // num_heads
    
    # Each core is responsible for a portion of the sequence length of Q, K, and V
    S_q_tiles = S_query // TILE_SIZE
    S_kv_tiles = S_kv // TILE_SIZE
    DH_tiles = DH // TILE_SIZE
    q_seq_per_core = math.ceil(S_q_tiles / (grid[0] * grid[1]))
    kv_seq_per_core = math.ceil(S_kv_tiles / (grid[0] * grid[1]))

    q_accessor = Stream(q_in, index_type=IndexType.TILE)
    kv_accessor = Stream(kv_in, index_type=IndexType.TILE)
    q_out_accessor = Stream(q_out, index_type=IndexType.TILE)
    k_out_accessor = Stream(k_out, index_type=IndexType.TILE)
    v_out_accessor = Stream(v_out, index_type=IndexType.TILE)
    
    # Size of a CB entry is DH_tiles pages, where page is a tile
    # NOTE: There's a question here. Should the shape of a CB be explicit? Should it be checked
    # when used in DM and compute kernels?
    # q_in_cb = CircularBuffer(q_accessor, shape=(DH_tiles,), buffer_factor=2)
    # kv_in_cb = CircularBuffer(kv_accessor, shape=(DH_tiles,), buffer_factor=2)
    # q_out_cb = CircularBuffer(q_out_accessor, shape=(DH_tiles,), buffer_factor=2)
    # k_out_cb = CircularBuffer(k_out_accessor, shape=(DH_tiles,), buffer_factor=2)
    # v_out_cb = CircularBuffer(v_out_accessor, shape=(DH_tiles,), buffer_factor=2)

    core_num = core_index() # core number in 2d grid
    start_q_seq_tile = core_num * q_seq_per_core
    end_q_seq_tile = min(start_q_seq_tile + q_seq_per_core, S_q_tiles)

    # Read-write Q
    for q_seq_tile in range(start_q_seq_tile, end_q_seq_tile):
        for batch in range(B):
            # NOTE: Normally you'd parallelize over batch as well but we aren't.
            for head in range(num_heads):
                read_index_slice = (
                    slice(batch, batch+1),
                    slice(q_seq_tile, q_seq_tile+1),
                    slice(head*DH_tiles, (head+1)*DH_tiles),
                )
                write_index_slice = (
                    slice(batch, batch+1),
                    slice(head, head+1),
                    slice(q_seq_tile, q_seq_tile+1),
                    slice(0, DH_tiles),
                )

                q_block = load(q_accessor[read_index_slice])

                # PETER: load codegened in dm0
                # q_in_cb.reserve()
                # tx = dma(q_accessor[read_index_slice], q_in_cb)
                # tx.wait()
                # q_in_cb.push()

                store(q_out_accessor[write_index_slice], q_block)

                # PETER: store codegened in dm1
                # q_in_cb.wait()
                # tx = dma(q_in_cb, q_out_accessor[write_index_slice])
                # tx.wait()
                # q_in_cb.pop()

    start_kv_seq_tile = core_num * kv_seq_per_core
    end_kv_seq_tile = min(start_kv_seq_tile + kv_seq_per_core, S_kv_tiles)
    
    # Read-write KV
    for tensor_idx in range(2): # K or V
        # If V tensor, offset is all heads of K, otherwise 0
        kv_offset = tensor_idx * num_heads * DH_tiles
        kv_out_accessor = k_out_accessor if tensor_idx == 0 else v_out_accessor

        for kv_seq_tile in range(start_kv_seq_tile, end_kv_seq_tile):
            for batch in range(B):
                # NOTE: Normally you'd parallelize over batch as well but we aren't.
                for head in range(num_heads):
                    read_index_slice = (
                        slice(batch, batch+1),
                        slice(kv_seq_tile, kv_seq_tile+1),
                        slice(kv_offset + head*DH_tiles, kv_offset + (head+1)*DH_tiles),
                    )

                    write_index_slice = (
                        slice(batch, batch+1),
                        slice(head, head+1),
                        slice(kv_seq_tile, kv_seq_tile+1),
                        slice(0, DH_tiles),
                    )

                    kv_block = load(kv_accessor[read_index_slice])

                    # PETER: load codegened in dm0
                    # kv_in_cb.reserve()
                    # tx = dma(kv_accessor[read_index_slice], kv_in_cb)
                    # tx.wait()
                    # kv_in_cb.push()

                    store(kv_out_accessor[write_index_slice], kv_block)

                    # PETER: store codegened in dm1
                    # kv_in_cb.wait()
                    # tx = dma(kv_in_cb, kv_out_accessor[write_index_slice])
                    # tx.wait()
                    # kv_in_cb.pop()

B, S_query, NH, D = 2, 4096, 8, 128
S_kv = 125 # purposefully unaligned, but tilized

q_in = torch.randn(B, S_query, NH*D)
kv_in = torch.randn(B, S_kv, 2*NH*D)

q_out = torch.zeros(B, NH, S_query, D)
k_out = torch.zeros(B, NH, S_kv, D)
v_out = torch.zeros(B, NH, S_kv, D)

create_qkv_heads(q_in, kv_in, q_out, k_out, v_out, num_heads=NH)

golden_q_out = q_in.view(B, S_query, NH, D).permute(0, 2, 1, 3)
golden_k_out, golden_v_out = kv_in.view(B, S_kv, 2, NH, D).permute(2, 0, 3, 1, 4).unbind(0)

assert_pcc(golden_q_out, q_out)
assert_pcc(golden_k_out, k_out)
assert_pcc(golden_v_out, v_out)
