# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pykernel.pykernel_ast import *
from pykernel.types import *


@ttkernel_noc_compile()
def reader_bmm_8bank_output_tiles_partitioned(
    cb_id_in0: CircularBuffer, cb_id_in1: CircularBuffer
):
    src0_addr = get_arg_val(int, 0)
    src1_addr = get_arg_val(int, 1)
    Mt = get_arg_val(int, 2)
    Kt = get_arg_val(int, 3)
    Nt = get_arg_val(int, 4)
    MtKt = get_arg_val(int, 5)
    KtNt = get_arg_val(int, 6)
    batch = get_arg_val(int, 7)
    bcast_B = get_arg_val(int, 8)
    output_tile_start_id = get_arg_val(int, 9)
    num_output_tiles = get_arg_val(int, 10)
    MtNt = get_arg_val(int, 11)

    src0_is_dram = get_compile_time_arg_val(0) == 1
    src1_is_dram = get_compile_time_arg_val(1) == 1

    itileA = output_tile_start_id / Nt * Kt

    onetile = 1
    in0_tile_bytes = get_tile_size(cb_id_in0)
    in0_data_format = get_dataformat(cb_id_in0)
    in1_tile_bytes = get_tile_size(cb_id_in1)
    in1_data_format = get_dataformat(cb_id_in1)

    # outbatch = output_tile_start_id % MtNt
    # itileB_batch = output_tile_start_id % Nt
    # itileB = itileB_batch

    s0 = get_interleaved_addr_gen_fast(
        src0_is_dram, src0_addr, in0_tile_bytes, in0_data_format
    )
    s1 = get_interleaved_addr_gen_fast(
        src1_is_dram, src1_addr, in1_tile_bytes, in1_data_format
    )

    l1_write_addr_in0 = get_write_ptr(cb_id_in0)
    noc_async_read_tile(itileA, s0, l1_write_addr_in0)
