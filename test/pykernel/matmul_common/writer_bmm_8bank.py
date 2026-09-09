# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel import ttkernel_noc_compile, CircularBuffer, Kernel


@ttkernel_noc_compile()
def writer_bmm_8bank(cb_id_out0: CircularBuffer):
    dst_addr = get_arg_val(int, 0)
    Mt = get_arg_val(int, 2)
    Nt = get_arg_val(int, 4)
    batch = get_arg_val(int, 7)

    onetile = 1
    tile_bytes = get_tile_size(cb_id_out0)
    itileC: int = 0

    tensor_accessor_args = TensorAccessorArgs(cta_base=1, crta_base=0)
    s = TensorAccessor(tensor_accessor_args, dst_addr, tile_bytes)

    for nb in range(0, batch, 1):
        for mt_C in range(0, Mt, 1):
            for nt_C in range(0, Nt, 1):
                cb_wait_front(cb_id_out0, onetile)
                l1_read_addr = get_read_ptr(cb_id_out0)
                noc_async_write_tile(itileC, s, l1_read_addr)
                noc_async_write_barrier()
                cb_pop_front(cb_id_out0, onetile)
                itileC += 1

    return


cb_in0 = CircularBuffer(0)
cb_in1 = CircularBuffer(1)
kernel_string = writer_bmm_8bank(cb_in0, cb_in1)
py_kernel = Kernel("writer_bmm_8bank", kernel_string)
py_kernel.dump()
