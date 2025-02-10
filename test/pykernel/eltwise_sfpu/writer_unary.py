# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *
from pykernel.types import *


@ttkernel_compile
def writer_unary(cb_in: CircularBuffer, cb_out: CircularBuffer):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%arg0: !ttkernel.cb<{{.*}}>, %arg1: !ttkernel.cb<{{.*}}>) {
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: %[[DST_ADDR:.*]] = memref.alloca(){{.*}}
    # CHECK: %[[BANK_ID:.*]] = "ttkernel.get_arg_val"{{.*}}
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    dst_addr: int = get_arg_val(int, 0)
    bank_id = get_arg_val(int, 1)
    num_tiles = get_arg_val(int, 2)

    # CHECK: {{.*}}"ttkernel.get_tile_size"{{.*}}
    ublock_size_bytes = get_tile_size(cb_out)
    ublock_size_tiles = 1

    for i in range(0, num_tiles, ublock_size_tiles):
        # CHECK: %[[DST_NOC_ADDR:.*]] = "ttkernel.get_noc_addr_from_bank_id"(%[[BANK_ID]],{{.*}}
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

        # CHECK: {{.*}}memref.load %[[DST_ADDR]]{{.*}}
        # CHECK: {{.*}}arith.addi{{.*}}
        # CHECK: memref.store {{.*}} %[[DST_ADDR]]{{.*}}
        dst_addr = dst_addr + ublock_size_bytes

    return


cb_in = CircularBuffer(0)
cb_out = CircularBuffer(16)
writer_unary(cb_in, cb_out)
