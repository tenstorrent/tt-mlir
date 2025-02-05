# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *
from pykernel.types import *


@ttkernel_compile
def reader_unary(cb_in: CircularBuffer, cb_out: CircularBuffer):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%arg0: !ttkernel.cb<{{.*}}>, %arg1: !ttkernel.cb<{{.*}}>) {
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: %[[SRC_ADDR:.*]] = memref.alloca(){{.*}}
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    src_addr: int = get_arg_val(int, 0)
    bank_id = get_arg_val(int, 1)
    num_tiles = get_arg_val(int, 2)

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

        # CHECK: {{.*}}memref.load %[[SRC_ADDR]]{{.*}}
        # CHECK: {{.*}}arith.addi{{.*}}
        # CHECK: memref.store {{.*}} %[[SRC_ADDR]]{{.*}}
        src_addr = src_addr + ublock_size_bytes

    return


cb_in = CircularBuffer(0)
cb_out = CircularBuffer(16)
reader_unary(cb_in, cb_out)
