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


class EltwiseSFPUPyKernelOp(PyKernelOp):
    # KERNEL DEFINITIONS
    @compute_thread()
    def eltwise_sfpu(
        cb_in: CircularBuffer,
        cb_out: CircularBuffer,
        per_core_block_cnt: CompileTimeValue,
        per_core_block_dim: CompileTimeValue,
    ):
        unary_op_init_common(cb_in, cb_out)
        for i in range(0, per_core_block_cnt, 1):
            cb_reserve_back(cb_out, per_core_block_dim)
            for j in range(0, per_core_block_dim, 1):
                tile_regs_acquire()
                cb_wait_front(cb_in, 1)

                copy_tile(cb_in, 0, 0)

                exp_tile_init()
                exp_tile(0)

                tile_regs_commit()
                tile_regs_wait()
                pack_tile(0, cb_out, 0)

                cb_pop_front(cb_in, 1)
                tile_regs_release()

            cb_push_back(cb_out, per_core_block_dim)
        return

    @writer_thread()
    def writer_unary_interleaved(
        cb_out: CircularBuffer,
        dst_addr,
        num_tiles,
        start_id,
    ):
        onetile = 1
        tile_bytes = get_tile_size(cb_out)

        tensor_accessor_args = TensorAccessorArgs(1, 0)
        s0 = TensorAccessor(tensor_accessor_args, dst_addr, tile_bytes)

        end_id = start_id + num_tiles
        ii: int = start_id
        for i in range(start_id, end_id, onetile):
            cb_wait_front(cb_out, onetile)
            l1_read_addr = get_read_ptr(cb_out)
            noc_async_write_tile(ii, s0, l1_read_addr)
            noc_async_write_barrier()
            cb_pop_front(cb_out, onetile)
            ii += onetile
        return

    @reader_thread()
    def reader_unary_interleaved(
        cb_in: CircularBuffer,
        src_addr,
        num_tiles,
        start_id,
    ):
        onetile = 1
        tile_bytes = get_tile_size(cb_in)

        tensor_accessor_args = TensorAccessorArgs(1, 0)
        s0 = TensorAccessor(tensor_accessor_args, src_addr, tile_bytes)

        end_id = start_id + num_tiles
        ii: int = start_id
        for i in range(start_id, end_id, onetile):
            cb_reserve_back(cb_in, onetile)
            l1_write_addr = get_write_ptr(cb_in)
            noc_async_read_tile(ii, s0, l1_write_addr)
            noc_async_read_barrier()
            cb_push_back(cb_in, onetile)
            ii += onetile
        return

    def invoke(
        self,  # super() has invoke signature (*tensors, **options)
        in_tensor,
        out_tensor,  # Tensor Definitions are positional args
    ):
        cb_in = self.create_cb(in_tensor, 0)
        cb_out = self.create_cb(out_tensor, 1)
        start_id = 0
        num_tiles = ceil(max(map(lambda t: t.volume(), [in_tensor, out_tensor])) / 1024)

        self.set_tensor_accessor_config([in_tensor, out_tensor])

        kernels = [
            self.create_kernel(
                EltwiseSFPUPyKernelOp.eltwise_sfpu,
                cb_in,
                cb_out,
                per_core_block_cnt=num_tiles,
                per_core_block_dim=1,
            ),
            self.create_kernel(
                EltwiseSFPUPyKernelOp.writer_unary_interleaved,
                cb_out,
                out_tensor.buffer_address(),
                num_tiles,
                start_id,
            ),
            self.create_kernel(
                EltwiseSFPUPyKernelOp.reader_unary_interleaved,
                cb_in,
                in_tensor.buffer_address(),
                num_tiles,
                start_id,
            ),
        ]

        return self.create_program(kernels, [cb_in, cb_out])


def main(device):
    # I/O Tensor Definitions
    num_tiles = 4
    shape = [1, num_tiles, 32, 32]
    data = torch.rand(shape).to(torch.bfloat16)

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )

    io_tensors = [input_tensor, output_tensor]

    # Define Custom Generic Op
    eltwise_exp_op = EltwiseSFPUPyKernelOp()

    # Run tests against the golden "exp" op.
    output = eltwise_exp_op(input_tensor, output_tensor)
    golden = ttnn.exp(input_tensor)

    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    print(f"input_tensor: {input_tensor}")
    print(f"torch_golden: {torch_golden}")
    print(f"torch_output: {torch_output}")

    matching = torch.allclose(torch_golden, torch_output)
    print(f"Tensors are matching: {matching}")
    assert matching


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    main(device)
    ttnn.close_device(device)
