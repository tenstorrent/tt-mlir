# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pykernel import (
    PyKernelOp,
    reader_thread,
    writer_thread,
    compute_thread,
    CircularBuffer,
)

from math import ceil

import ttnn
import torch


class MatmulMulticorePyKernelOp(PyKernelOp):
    def __init__(self, max_core_ranges=None):
        super().__init__()
        self.max_core_ranges = max_core_ranges

    # KERNEL DEFINITIONS
    @compute_thread()
    def matmul_multicore(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
        num_output_tiles,
        Kt,
    ):
        mm_init(cb_in0, cb_in1, cb_out, 0)

        for i in range(0, num_output_tiles, 1):
            tile_regs_acquire()  # acquire lock on DST reg for MATH
            for kt in range(0, Kt, 1):
                cb_wait_front(cb_in0, 1)
                cb_wait_front(cb_in1, 1)

                matmul_tiles(cb_in0, cb_in1, 0, 0, 0, 0)

                cb_pop_front(cb_in0, 1)
                cb_pop_front(cb_in1, 1)

            cb_reserve_back(cb_out, 1)
            tile_regs_commit()  # release lock on DST reg for MATH

            tile_regs_wait()  # acquire lock on DST reg for PACK
            pack_tile(0, cb_out, 0)
            cb_push_back(cb_out, 1)
            tile_regs_release()  # release lock on DST reg for PACK

        return

    @reader_thread()
    def reader_multicore_matmul(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        src_addr0,
        src_addr1,
        Mt,
        Kt,
        Nt,
        start_id,
        num_tiles,
    ):
        in0_tile_bytes = get_tile_size(cb_in0)
        tensor_accessor_args = TensorAccessorArgs(2, 0)
        addr_gen_a = TensorAccessor(tensor_accessor_args, src_addr0, in0_tile_bytes)

        in1_tile_bytes = get_tile_size(cb_in1)
        tensor_accessor_args = TensorAccessorArgs(2, 0)
        addr_gen_b = TensorAccessor(tensor_accessor_args, src_addr1, in1_tile_bytes)

        for output_tile in range(0, num_tiles, 1):
            current_tile_id = start_id + output_tile
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt

            for k in range(0, Kt, 1):
                tileA = out_row * Kt + k
                cb_reserve_back(cb_in0, 1)
                l1_write_addr_in0 = get_write_ptr(cb_in0)
                noc_async_read_tile(tileA, addr_gen_a, l1_write_addr_in0)
                noc_async_read_barrier()
                cb_push_back(cb_in0, 1)

                tileB = k * Nt + out_col
                cb_reserve_back(cb_in1, 1)
                l1_write_addr_in1 = get_write_ptr(cb_in1)
                noc_async_read_tile(tileB, addr_gen_b, l1_write_addr_in1)
                noc_async_read_barrier()
                cb_push_back(cb_in1, 1)

        return

    @writer_thread()
    def writer_multicore_matmul(
        cb_out: CircularBuffer,
        dst_addr,
        num_tiles,
        start_id,
    ):
        tile_bytes = get_tile_size(cb_out)
        tensor_accessor_args = TensorAccessorArgs(1, 0)
        addr_gen_c = TensorAccessor(tensor_accessor_args, dst_addr, tile_bytes)

        end_id = start_id + num_tiles
        for i in range(start_id, end_id, 1):
            cb_wait_front(cb_out, 1)
            l1_read_addr = get_read_ptr(cb_out)
            noc_async_write_tile(i, addr_gen_c, l1_read_addr)
            noc_async_write_barrier()
            cb_pop_front(cb_out, 1)

        return

    def define_core_ranges(self, tensors, options):
        core_0 = ttnn.CoreCoord(0, 0)
        if self.max_core_ranges is None:
            core_1 = ttnn.CoreCoord(1, 1)
        else:
            core_1 = self.max_core_ranges
        return ttnn.CoreRangeSet([ttnn.CoreRange(core_0, core_1)])

    def invoke(
        self,
        a_tensor,
        b_tensor,
        out_tensor,
    ):
        cb_in0 = self.create_cb(a_tensor, 0)
        cb_in1 = self.create_cb(b_tensor, 1)
        cb_out = self.create_cb(out_tensor, 2)
        start_id = 0

        self.set_tensor_accessor_config(a_tensor)

        Mt = a_tensor.shape[0] // 32
        Kt = a_tensor.shape[1] // 32
        Nt = b_tensor.shape[1] // 32

        num_output_tiles = Mt * Nt

        num_cores = self.get_core_ranges().num_cores()
        num_tiles_per_core = int(num_output_tiles / num_cores)

        if num_output_tiles % num_cores != 0:
            raise Exception("Uneven distribution of work not supported")

        # Define the multicore runtime arguments
        start_id_multicore = []

        # Go row-wise
        bb = self.get_core_ranges().bounding_box()
        for i in range(bb.start.x, bb.end.x + 1):
            start_id_multicore.append([])
            for j in range(bb.start.y, bb.end.y + 1):
                # Set for each core
                start_id_multicore[-1].append([start_id])
                start_id += num_tiles_per_core

        kernels = [
            self.create_kernel(
                MatmulMulticorePyKernelOp.matmul_multicore,
                cb_in0,
                cb_in1,
                cb_out,
                num_tiles_per_core,
                Kt,
            ),
            self.create_kernel(
                MatmulMulticorePyKernelOp.reader_multicore_matmul,
                cb_in0,
                cb_in1,
                a_tensor.buffer_address(),
                b_tensor.buffer_address(),
                Mt,
                Kt,
                Nt,
                start_id_multicore,
                num_tiles_per_core,
            ),
            self.create_kernel(
                MatmulMulticorePyKernelOp.writer_multicore_matmul,
                cb_out,
                out_tensor.buffer_address(),
                num_tiles_per_core,
                start_id_multicore,
            ),
        ]

        return self.create_program(kernels, [cb_in0, cb_in1, cb_out])


def main(device):
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

    multicore_matmul_op = MatmulMulticorePyKernelOp()

    output = multicore_matmul_op(a_tensor, b_tensor, output_tensor)
    golden = ttnn.matmul(a_tensor, b_tensor)

    torch_output = ttnn.to_torch(output)
    torch_golden = ttnn.to_torch(golden)

    print(f"a_tensor: {a_tensor}")
    print(f"b_tensor: {b_tensor}")
    print(f"torch_golden: {torch_golden}")
    print(f"torch_output: {torch_output}")

    matching = torch.allclose(torch_golden, torch_output, atol=0.75)
    print(f"Tensors are matching: {matching}")
    assert matching


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    main(device)
    ttnn.close_device(device)
