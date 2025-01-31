# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# #include <stdint.h>

# #include "dataflow_api.h"

# void kernel_main() {
#     uint32_t src_addr  = get_arg_val<uint32_t>(0);
#     uint32_t bank_id = get_arg_val<uint32_t>(1);
#     uint32_t num_tiles = get_arg_val<uint32_t>(2);

#     constexpr uint32_t cb_id_in0 = 0;

#     // ublocks size defined in tiles
#     constexpr uint32_t ublock_size_tiles = 1;
#     uint32_t ublock_size_bytes = get_tile_size(cb_id_in0) * ublock_size_tiles;

#     // read a ublock of tiles from src to CB, and then push the ublock to unpacker
#     for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
#         uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_addr);

#         cb_reserve_back(cb_id_in0, ublock_size_tiles);
#         uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
#         noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes);

#         noc_async_read_barrier();

#         cb_push_back(cb_id_in0, ublock_size_tiles);
#         src_addr += ublock_size_bytes;
#     }
# }

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *


@ttkernel_compile
def reader_unary(cb_in: int, cb_out: int):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%[[arg0:.*]]: !ttkernel.cb<{{.*}}>, %[[arg1:.*]]: !ttkernel.cb<{{.*}}>) {
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    src_addr = get_arg_val(type_int, 0)
    bank_id = get_arg_val(type_int, 1)
    num_tiles = get_arg_val(type_int, 2)

    # CHECK: {{.*}}"ttkernel.get_tile_size"{{.*}}
    ublock_size_tiles = 1
    ublock_size_bytes = get_tile_size(cb_in) * ublock_size_tiles

    for i in range(0, num_tiles, ublock_size_tiles):
        # CHECK: %[[SRC_NOC_ADDR:.*]] = "ttkernel.get_noc_addr_from_bank_id"{{.*}}
        src_noc_addr = get_noc_addr_from_bank_id(bank_id, src_addr)

        # CHECK: "ttkernel.cb_reserve_back"{{.*}}
        cb_reserve_back(cb_in, ublock_size_tiles)

        # CHECK: {{.*}}"ttkernel.get_write_ptr"{{.*}}
        # CHECK: "ttkernel.noc_async_read"(%[[SRC_NOC_ADDR]], {{.*}}, {{.*}}){{.*}}
        # CHECK: "ttkernel.noc_async_read_barrier"{{.*}}
        l1_write_addr = get_write_ptr(cb_in)
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes)
        noc_async_read_barrier()

        # CHECK: "ttkernel.cb_push_back"{{.*}}
        cb_push_back(cb_in, ublock_size_tiles)

        src_addr = src_addr + ublock_size_bytes

    return


reader_unary(0, 16)
