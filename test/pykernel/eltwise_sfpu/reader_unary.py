# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *
from pykernel.types import *


@ttkernel_noc_compile(verbose=True)
def reader_unary(cb_in: CircularBuffer, cb_out: CircularBuffer, rt_args):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%arg0: !ttkernel.cb<{{.*}}>, %arg1: !ttkernel.cb<{{.*}}>) {
    # CHECK: emitc.verbatim "// --- Python Function {{.*}}"
    # CHECK: emitc.verbatim "// src_addr: int = rt_args[0]"
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: %[[SRC_ADDR:.*]] = memref.alloca(){{.*}}
    # CHECK: emitc.verbatim "// bank_id, num_tiles = rt_args[1:3]"
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    # CHECK: {{.*}}"ttkernel.get_arg_val"{{.*}}
    src_addr: int = rt_args[0]
    bank_id, num_tiles = rt_args[1:3]

    # CHECK: emitc.verbatim "// ublock_size_tiles = 1"
    # CHECK: emitc.verbatim "// ublock_size_bytes = get_tile_size(cb_in) * ublock_size_tiles"
    # CHECK: {{.*}}"ttkernel.get_tile_size"{{.*}}
    ublock_size_tiles = 1
    ublock_size_bytes = get_tile_size(cb_in) * ublock_size_tiles

    # CHECK: emitc.verbatim "// for i in range(0, num_tiles, ublock_size_tiles):"
    for i in range(0, num_tiles, ublock_size_tiles):
        # CHECK: emitc.verbatim "// src_noc_addr = get_noc_addr_from_bank_id(bank_id, src_addr)"
        # CHECK: %[[SRC_NOC_ADDR:.*]] = "ttkernel.get_noc_addr_from_bank_id"{{.*}}
        src_noc_addr = get_noc_addr_from_bank_id(bank_id, src_addr)

        # CHECK: "ttkernel.cb_reserve_back"{{.*}}
        cb_reserve_back(cb_in, ublock_size_tiles)

        # CHECK: emitc.verbatim "// l1_write_addr = get_write_ptr(cb_in)"
        # CHECK: {{.*}}"ttkernel.get_write_ptr"{{.*}}
        # CHECK: "ttkernel.noc_async_read"(%[[SRC_NOC_ADDR]], {{.*}}, {{.*}}){{.*}}
        # CHECK: "ttkernel.noc_async_read_barrier"{{.*}}
        l1_write_addr = get_write_ptr(cb_in)
        noc_async_read(src_noc_addr, l1_write_addr, ublock_size_bytes)
        noc_async_read_barrier()

        # CHECK: "ttkernel.cb_push_back"{{.*}}
        cb_push_back(cb_in, ublock_size_tiles)

        # CHECK: emitc.verbatim "// src_addr = src_addr + ublock_size_bytes"
        # CHECK: {{.*}}memref.load %[[SRC_ADDR]]{{.*}}
        # CHECK: {{.*}}arith.addi{{.*}}
        # CHECK: memref.store {{.*}} %[[SRC_ADDR]]{{.*}}
        src_addr = src_addr + ublock_size_bytes

    return


cb_in = CircularBuffer(0)
cb_out = CircularBuffer(16)
kernel_string = reader_unary(cb_in, cb_out)
py_kernel = Kernel("reader_unary", kernel_string)
py_kernel.dump()
