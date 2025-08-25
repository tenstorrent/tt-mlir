# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel import ttkernel_tensix_compile, CircularBuffer, Kernel


@ttkernel_tensix_compile()
def add_2_tiles(cb_in0: CircularBuffer, cb_in1: CircularBuffer, cb_out: CircularBuffer):
    binary_op_init_common(cb_in0, cb_in1, cb_out)
    add_tiles_init(cb_in0, cb_in1)

    cb_wait_front(cb_in0, 1)
    cb_wait_front(cb_in1, 1)

    tile_regs_acquire()

    add_tiles(cb_in0, cb_in1, 0, 0, 0)

    tile_regs_commit()

    tile_regs_wait()
    pack_tile(0, cb_out, 0)
    tile_regs_release()

    cb_pop_front(cb_in0, 1)
    cb_pop_front(cb_in1, 1)

    cb_push_back(cb_out, 1)

    return


cb_in0 = CircularBuffer(0)
cb_in1 = CircularBuffer(1)
cb_out = CircularBuffer(16)
kernel_string = add_2_tiles(cb_in0, cb_in1, cb_out)
py_kernel = Kernel("add_2_tiles", kernel_string)
py_kernel.dump()
