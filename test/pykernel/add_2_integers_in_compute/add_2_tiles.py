# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *
from pykernel.types import *


@ttkernel_compile
def add_2_tiles(cb_in0: CircularBuffer, cb_in1: CircularBuffer, cb_out: CircularBuffer):
    binary_op_init_common(cb_in0, cb_in1, cb_out)
    add_tiles_init(cb_in0, cb_in1)

    # wait for a block of tiles in each of input CBs
    cb_wait_front(cb_in0, 1)
    cb_wait_front(cb_in1, 1)

    tile_regs_acquire()  # acquire 8 tile registers

    add_tiles(cb_in0, cb_in1, 0, 0, 0)

    tile_regs_commit()  # signal the packer

    tile_regs_wait()  # packer waits here
    pack_tile(0, cb_out, 0)
    tile_regs_release()  # packer releases

    cb_pop_front(cb_in0, 1)
    cb_pop_front(cb_in1, 1)

    cb_push_back(cb_out, 1)

    return


cb_in0 = CircularBuffer(0)
cb_in1 = CircularBuffer(1)
cb_out = CircularBuffer(16)
add_2_tiles(cb_in0, cb_in1, cb_out)
