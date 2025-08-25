# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pykernel import (
    PyKernelOp,
    reader_thread,
    writer_thread,
    compute_thread,
    CircularBuffer,
    CompileTimeValue,
)

from math import ceil

import ttnn
import torch


class MatmulSinglecorePyKernelOp(PyKernelOp):
    def __init__(self):
        super().__init__()

    # KERNEL DEFINITIONS
    @compute_thread(optimize=True)
    def mm(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
        M: CompileTimeValue,
        N: CompileTimeValue,
        K: CompileTimeValue,
    ):
        mm_init(cb_in0, cb_in1, cb_out, 0)

        for i in range(0, M, 1):
            for j in range(0, N, 1):
                tile_regs_acquire()
                for k in range(0, K, 1):
                    cb_wait_front(cb_in0, 1)
                    cb_wait_front(cb_in1, 1)

                    matmul_tiles(cb_in0, cb_in1, 0, 0, 0, 0)

                    cb_pop_front(cb_in0, 1)
                    cb_pop_front(cb_in1, 1)
                cb_reserve_back(cb_out, 1)
                tile_regs_commit()
                tile_regs_wait()
                pack_tile(0, cb_out, 0)
                cb_push_back(cb_out, 1)
                tile_regs_release()
        return

    @writer_thread(optimize=True)
    def writer_single_core_mm(
        cb_out: CircularBuffer,
        dst_addr,
        M,
        N,
    ):
        tile_bytes = get_tile_size(cb_out)

        tensor_accessor_args = TensorAccessorArgs(1, 0)
        s0 = TensorAccessor(tensor_accessor_args, dst_addr, tile_bytes)

        for m in range(0, M, 1):
            for n in range(0, N, 1):
                # Write result to Output Tensor
                cb_wait_front(cb_out, 1)
                l1_read_addr = get_read_ptr(cb_out)
                tile_idx = m * N + n
                noc_async_write_tile(tile_idx, s0, l1_read_addr)
                noc_async_write_barrier()
                cb_pop_front(cb_out, 1)

        return

    @reader_thread(optimize=True)
    def reader_single_core_mm(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        src_addr0,
        src_addr1,
        M,
        N,
        K,
    ):
        tile_bytes0 = get_tile_size(cb_in0)
        tensor_accessor_args = TensorAccessorArgs(2, 0)
        s0 = TensorAccessor(tensor_accessor_args, src_addr0, tile_bytes0)

        tile_bytes1 = get_tile_size(cb_in1)
        tensor_accessor_args = TensorAccessorArgs(2, 0)
        s1 = TensorAccessor(tensor_accessor_args, src_addr1, tile_bytes1)

        for m in range(0, M, 1):
            for n in range(0, N, 1):
                for k in range(0, K, 1):
                    # Read tile from SRC0 at (m, k)
                    a_tile_idx = m * K + k
                    cb_reserve_back(cb_in0, 1)
                    src0_write_addr = get_write_ptr(cb_in0)
                    noc_async_read_tile(a_tile_idx, s0, src0_write_addr)
                    noc_async_read_barrier()
                    cb_push_back(cb_in0, 1)

                    # Read tile from SRC1 at (k, n)
                    b_tile_idx = k * N + n
                    cb_reserve_back(cb_in1, 1)
                    src1_write_addr = get_write_ptr(cb_in1)
                    noc_async_read_tile(b_tile_idx, s1, src1_write_addr)
                    noc_async_read_barrier()
                    cb_push_back(cb_in1, 1)

        return

    def invoke(
        self,  # super() has invoke signature (*tensors, **options)
        a_tensor,
        b_tensor,
        out_tensor,  # Tensor Definitions are positional args
    ):
        cb_in0 = self.create_cb(a_tensor, 0)
        cb_in1 = self.create_cb(b_tensor, 1)
        cb_out = self.create_cb(out_tensor, 16)

        self.set_tensor_accessor_config([a_tensor, b_tensor, out_tensor])

        # Calculate M, N, K as tile numbers, tiles are 32x32
        # A[MxK], B[KxN], Output[MxN]
        M = a_tensor.shape[0] // 32
        K = a_tensor.shape[1] // 32
        N = b_tensor.shape[1] // 32

        kernels = [
            self.create_kernel(
                MatmulSinglecorePyKernelOp.mm, cb_in0, cb_in1, cb_out, M=M, N=N, K=K
            ),
            self.create_kernel(
                MatmulSinglecorePyKernelOp.writer_single_core_mm,
                cb_out,
                out_tensor.buffer_address(),
                M,
                N,
            ),
            self.create_kernel(
                MatmulSinglecorePyKernelOp.reader_single_core_mm,
                cb_in0,
                cb_in1,
                a_tensor.buffer_address(),
                b_tensor.buffer_address(),
                M,
                N,
                K,
            ),
        ]

        return self.create_program(kernels, [cb_in0, cb_in1, cb_out])


def main(device):
    # I/O Tensor Definitions

    # Given two matrices being inputted, MxK and KxN, the resultant matrix will be of MxN dimensions.
    M = 4
    K = 4
    N = 4

    a_shape = [M * 32, K * 32]
    a_data = torch.rand(a_shape).to(torch.bfloat16)

    b_shape = [K * 32, N * 32]
    b_data = torch.rand(b_shape).to(torch.bfloat16)

    out_shape = [M * 32, N * 32]

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    a_tensor = ttnn.from_torch(
        a_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    b_tensor = ttnn.from_torch(
        b_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )

    matmul_op = MatmulSinglecorePyKernelOp()

    # Run tests against the golden "add" op.
    output = matmul_op(a_tensor, b_tensor, output_tensor)
    golden = ttnn.matmul(a_tensor, b_tensor)

    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    print(f"a_tensor: {a_tensor}")
    print(f"b_tensor: {b_tensor}")
    print(f"torch_golden: {torch_golden}")
    print(f"torch_output: {torch_output}")

    # Accuracy errors due to device flags that we may not be setting and using (which ttnn could be using)
    matching = torch.allclose(torch_golden, torch_output, atol=1)
    print(f"Tensors are matching: {matching}")
    assert matching


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    main(device)
    ttnn.close_device(device)
