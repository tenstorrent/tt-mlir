# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel import ttkernel_noc_compile, CircularBuffer, Kernel


@ttkernel_noc_compile()
def write_1_tile(cb_out: CircularBuffer):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%arg0: !ttkernel.cb<{{.*}}>, %arg1: !ttkernel.cb<{{.*}}>) {
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    dst_addr = get_arg_val(int, 0)
    dst_bank_id = get_arg_val(int, 0)

    dst_noc_addr = get_noc_addr_from_bank_id(dst_bank_id, dst_addr)

    ublock_size_bytes = get_tile_size(cb_out)
    l1_read_addr = get_read_ptr(cb_out)

    cb_wait_front(cb_out, 1)
    noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes)
    noc_async_write_barrier()
    cb_pop_front(cb_out, 1)

    return


cb_out = CircularBuffer(16)
kernel_string = write_1_tile(cb_out)
py_kernel = Kernel("write_1_tile", kernel_string)
py_kernel.dump()
