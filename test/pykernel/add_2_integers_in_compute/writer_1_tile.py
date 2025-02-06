# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *
from pykernel.types import *


@ttkernel_noc_compile()
def write_1_tile(cb_out: CircularBuffer):
    dst_addr = get_arg_val(int, 0)
    dst_bank_id = get_arg_val(int, 1)

    dst_noc_addr = get_noc_addr_from_bank_id(dst_bank_id, dst_addr)

    ublock_size_bytes = get_tile_size(cb_out)
    l1_read_addr = get_read_ptr(cb_out)

    cb_wait_front(cb_out, 1)
    noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes)
    noc_async_write_barrier()
    cb_pop_front(cb_out, 1)

    return


cb_out = CircularBuffer(16)
write_1_tile(cb_out)
