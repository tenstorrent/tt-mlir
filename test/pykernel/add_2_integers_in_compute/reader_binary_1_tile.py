# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel import ttkernel_noc_compile, CircularBuffer, Kernel


@ttkernel_noc_compile()
def reader_binary_1_tile(cb_in0: CircularBuffer, cb_in1: CircularBuffer):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%arg0: !ttkernel.cb<{{.*}}>, %arg1: !ttkernel.cb<{{.*}}>) {
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
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
kernel_string = reader_binary_1_tile(cb_in0, cb_in1)
py_kernel = Kernel("reader_binary_1_tile", kernel_string)
py_kernel.dump()
