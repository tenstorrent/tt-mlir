# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# REQUIRES: pykernel

from pykernel import ttkernel_noc_compile, CircularBuffer, Kernel


@ttkernel_noc_compile()
def reader_bmm_8bank(cb_id_in0: CircularBuffer, cb_id_in1: CircularBuffer):
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
    src0_addr = get_arg_val(int, 0)
    src1_addr = get_arg_val(int, 1)
    Mt = get_arg_val(int, 2)
    Kt = get_arg_val(int, 3)
    Nt = get_arg_val(int, 4)
    MtKt = get_arg_val(int, 5)
    KtNt = get_arg_val(int, 6)
    batch = get_arg_val(int, 7)
    bcast_B = get_arg_val(int, 8)

    src0_is_dram = get_compile_time_arg_val(int, 0) == 1
    src1_is_dram = get_compile_time_arg_val(int, 1) == 1

    onetile = 1
    src0_tile_bytes = get_tile_size(cb_id_in0)
    src0_data_format = get_dataformat(cb_id_in0)
    src1_tile_bytes = get_tile_size(cb_id_in1)
    src1_data_format = get_dataformat(cb_id_in1)

    itileA_batch: int = 0
    itileB_batch: int = 0

    s0 = get_interleaved_addr_gen_fast(
        src0_is_dram, src0_addr, src0_tile_bytes, src0_data_format
    )
    s1 = get_interleaved_addr_gen_fast(
        src1_is_dram, src1_addr, src1_tile_bytes, src1_data_format
    )

    for nb in range(0, batch, 1):
        itileA: int = itileA_batch + 0
        for mt in range(0, Mt, 1):
            itileB: int = itileB_batch + 0
            for nt in range(0, Nt, 1):
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

                    itileA += 1
                    itileB += Nt

                itileB -= KtNt + 1
                itileA -= Kt
            itileA += Kt
        itileA_batch += MtKt
        if bcast_B == 0:
            itileB_batch += KtNt

    return


cb_in0 = CircularBuffer(0)
cb_in1 = CircularBuffer(1)
kernel_string = reader_bmm_8bank(cb_in0, cb_in1)
py_kernel = Kernel("reader_bmm_8bank", kernel_string)
py_kernel.dump()
