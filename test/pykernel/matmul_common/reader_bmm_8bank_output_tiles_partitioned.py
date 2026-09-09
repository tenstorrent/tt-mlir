# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel import ttkernel_noc_compile, CircularBuffer, Kernel


@ttkernel_noc_compile()
def reader_bmm_8bank_output_tiles_partitioned(
    cb_id_in0: CircularBuffer, cb_id_in1: CircularBuffer
):
    # CHECK: module {
    # CHECK: func.func @{{.*}}(%arg0: !ttkernel.cb<{{.*}}>, %arg1: !ttkernel.cb<{{.*}}>) {
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    # CHECK: {{.*}}ttkernel.get_arg_val{{.*}}
    src0_addr = get_arg_val(int, 0)
    src1_addr = get_arg_val(int, 1)
    Mt = get_arg_val(int, 2)
    Kt = get_arg_val(int, 3)
    Nt = get_arg_val(int, 4)
    MtKt = get_arg_val(int, 5)
    KtNt = get_arg_val(int, 6)
    batch = get_arg_val(int, 7)
    bcast_B = get_arg_val(int, 8)
    output_tile_start_id = get_arg_val(int, 9)
    num_output_tiles = get_arg_val(int, 10)
    MtNt = get_arg_val(int, 11)

    itileA: int = output_tile_start_id  # should be: output_tile_start_id / Nt * Kt

    onetile = 1
    in0_tile_bytes = get_tile_size(cb_id_in0)
    in1_tile_bytes = get_tile_size(cb_id_in1)

    outbatch: int = (
        output_tile_start_id * MtNt
    )  # should be: output_tile_start_id % MtNt
    itileB_batch: int = (
        output_tile_start_id * Nt
    )  # should be: output_tile_start_id % Nt
    itileB: int = itileB_batch + 0

    tensor_accessor_args = TensorAccessorArgs(cta_base=2, crta_base=0)
    s0 = TensorAccessor(tensor_accessor_args, src0_addr, in0_tile_bytes)
    tensor_accessor_args = TensorAccessorArgs(cta_base=3, crta_base=0)
    s1 = TensorAccessor(tensor_accessor_args, src1_addr, in1_tile_bytes)

    for n in range(0, num_output_tiles, 1):
        for kt in range(0, Kt, 1):
            cb_reserve_back(cb_id_in0, onetile)
            l1_write_addr_in0 = get_write_ptr(cb_id_in0)
            noc_async_read_tile(itileA, s0, l1_write_addr_in0)
            noc_async_read_barrier()
            cb_push_back(cb_id_in0, onetile)

            cb_reserve_back(cb_id_in1, onetile)
            l1_write_addr_in1 = get_write_ptr(cb_id_in1)
            noc_async_read_tile(itileB, s1, l1_write_addr_in1)
            noc_async_read_barrier()
            cb_push_back(cb_id_in1, onetile)

            itileA = itileA + 1
            itileB = itileB + Nt

        outbatch = outbatch + 1
        itileB_batch = itileB_batch + 1
        itileB = itileB - KtNt
        itileB = itileB + 1

        if itileB_batch == Nt:
            itileB_batch = 0
            itileB = itileB - Nt
            if outbatch == MtNt:
                if bcast_B == 0:
                    itileB = itileB + KtNt
                outbatch = 0
        else:
            itileA = itileA - Kt

    return


cb_in0 = CircularBuffer(0)
cb_in1 = CircularBuffer(1)
kernel_string = reader_bmm_8bank_output_tiles_partitioned(cb_in0, cb_in1)
py_kernel = Kernel("reader_bmm_8bank_output_tiles_partitioned", kernel_string)
py_kernel.dump()
