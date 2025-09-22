# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel import ttkernel_tensix_compile, CircularBuffer, Kernel


@ttkernel_tensix_compile()
def eltwise_sfpu(cb_in: CircularBuffer, cb_out: CircularBuffer):
    # CHECK: module {
    # CHECK: func.func @eltwise_sfpu() attributes {ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    # CHECK: {{.*}}ttkernel.get_compile_time_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_compile_time_arg_val{{.*}}
    per_core_block_cnt = get_compile_time_arg_val(int, 2)
    per_core_block_dim = get_compile_time_arg_val(int, 3)

    # CHECK: ttkernel.unary_op_init_common({{.*}})
    unary_op_init_common(cb_in, cb_out)

    for i in range(0, per_core_block_cnt, 1):
        # CHECK: ttkernel.cb_reserve_back{{.*}}
        cb_reserve_back(cb_out, per_core_block_dim)
        for j in range(0, per_core_block_dim, 1):
            # CHECK: ttkernel.tile_regs_acquire(){{.*}}
            # CHECK: ttkernel.cb_wait_front{{.*}}
            tile_regs_acquire()
            cb_wait_front(cb_in, 1)

            # CHECK: ttkernel.copy_tile{{.*}}
            # CHECK: ttkernel.pack_tile{{.*}}
            copy_tile(cb_in, 0, 0)
            pack_tile(0, cb_out, 0)
            # last parameter is a optional index in C++ but not optional in pybind

            # CHECK: ttkernel.cb_pop_front{{.*}}
            # CHECK: ttkernel.tile_regs_release(){{.*}}
            cb_pop_front(cb_in, 1)
            tile_regs_release()

        # CHECK: ttkernel.cb_push_back{{.*}}
        cb_push_back(cb_out, per_core_block_dim)

    return


cb_in = CircularBuffer(0)
cb_out = CircularBuffer(16)
kernel_string = eltwise_sfpu(cb_in, cb_out)
py_kernel = Kernel("eltwise_sfpu", kernel_string)
py_kernel.dump()
