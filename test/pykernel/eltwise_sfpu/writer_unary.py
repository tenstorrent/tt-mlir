# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *


@ttkernel_compile
def writer_unary(cb_in: int, cb_out: int):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%[[arg0:.*]]: !ttkernel.cb<{{.*}}>, %[[arg1:.*]]: !ttkernel.cb<{{.*}}>) {
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    dst_addr: int = get_arg_val(int, 0)
    bank_id = get_arg_val(int, 1)
    num_tiles = get_arg_val(int, 2)

    # CHECK: {{.*}}"ttkernel.get_tile_size"{{.*}}
    ublock_size_bytes = get_tile_size(cb_out)
    ublock_size_tiles = 1

    cb_id_out0 = 16

    for i in range(0, num_tiles, ublock_size_tiles):
        # CHECK: %[[DST_NOC_ADDR:.*]] = "ttkernel.get_noc_addr_from_bank_id"{{.*}}
        dst_noc_addr = get_noc_addr_from_bank_id(bank_id, dst_addr)

        # CHECK: "ttkernel.cb_wait_front"{{.*}}
        # CHECK: {{.*}}"ttkernel.get_read_ptr"{{.*}}
        cb_wait_front(cb_out, ublock_size_tiles)
        l1_read_addr = get_read_ptr(cb_out)

        # CHECK: "ttkernel.noc_async_write"({{.*}}, %[[DST_NOC_ADDR]], {{.*}}){{.*}}
        # CHECK: "ttkernel.noc_async_write_barrier"{{.*}}
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes)
        noc_async_write_barrier()

        # CHECK: "ttkernel.cb_pop_front"{{.*}}
        cb_pop_front(cb_out, ublock_size_tiles)
        dst_addr = dst_addr + ublock_size_bytes

    return


writer_unary(0, 16)
