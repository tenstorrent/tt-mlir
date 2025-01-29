# // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# //
# // SPDX-License-Identifier: Apache-2.0

# #include "dataflow_api.h"
# #include "debug/dprint.h"

# void kernel_main() {
#     uint32_t dst_addr  = get_arg_val<uint32_t>(0);
#     uint32_t bank_id = get_arg_val<uint32_t>(1);
#     uint32_t num_tiles = get_arg_val<uint32_t>(2);

#     constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

#     // single-tile ublocks
#     uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
#     uint32_t ublock_size_tiles = 1;

#     for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
#         uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_addr);

#         cb_wait_front(cb_id_out0, ublock_size_tiles);
#         uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
#         noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);

#         noc_async_write_barrier();

#         cb_pop_front(cb_id_out0, ublock_size_tiles);
#         dst_addr += ublock_size_bytes;
#     }
# }

# CBIndex is just uint8_t

from pykernel.pykernel_ast import *


@ttkernel_compile
def write_unary(cb_in, cb_out):
    dst_addr = get_arg_val(type_int, 0)
    bank_id = get_arg_val(type_int, 1)
    num_tiles = get_arg_val(type_int, 2)

    ublock_size_bytes = 5  # get_tile_size(16)                                   # get_tile_size not implemented
    ublock_size_tiles = 1

    cb_id_out0 = 16

    for i in range(0, num_tiles, ublock_size_tiles):
        # dst_noc_addr = get_noc_addr_from_bank_id(True, bank_id, dst_addr)   # get_noc_addr_from_bank_id not implemented

        cb_wait_front(cb_out, ublock_size_tiles)
        # l1_read_addr = get_read_ptr(cb_out)                                     # get_read_ptr not implemented - returns uint32_t
        # noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes)

        noc_async_write_barrier()

        cb_pop_front(cb_out, ublock_size_tiles)
        dst_addr += ublock_size_bytes
