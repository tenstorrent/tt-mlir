# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *
from pykernel.types import *


@ttkernel_noc_compile()
def reader_binary_1_tile(cb_in0: CircularBuffer, cb_in1: CircularBuffer):
    src0_addr = get_arg_val(int, 0)
    src1_addr = get_arg_val(int, 1)
    src0_bank_id = get_arg_val(int, 2)
    src1_bank_id = get_arg_val(int, 3)

    src0_noc_addr = get_noc_addr_from_bank_id(src0_bank_id, src0_addr)
    src1_noc_addr = get_noc_addr_from_bank_id(src1_bank_id, src1_addr)

    ublock_size_bytes_0 = get_tile_size(cb_in0)
    ublock_size_bytes_1 = get_tile_size(cb_in1)

    l1_write_addr_in0 = get_write_ptr(cb_in0)
    l1_write_addr_in1 = get_write_ptr(cb_in1)

    cb_reserve_back(cb_in0, 1)
    noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0)
    noc_async_read_barrier()
    cb_push_back(cb_in0, 1)

    cb_reserve_back(cb_in1, 1)
    noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1)
    noc_async_read_barrier()
    cb_push_back(cb_in1, 1)

    return


cb_in0 = CircularBuffer(0)
cb_in1 = CircularBuffer(1)
kernel_file = reader_binary_1_tile(cb_in0, cb_in1)
reader_binary_1_tile_kernel = Kernel(kernel_file)
reader_binary_1_tile_kernel.dump()
