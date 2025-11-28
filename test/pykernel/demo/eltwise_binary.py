# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

import ttnn
import torch

class EltwiseBinaryPyKernelOp(PyKernelOp):
    @compute_thread()
    def eltwise_binary_tiles_add(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
        num_tiles,
    ):
        dst_reg = 0
        binary_op_init_common(cb_in0, cb_in1, cb_out)
        add_tiles_init(cb_in0, cb_in1)

        for i in range(0, num_tiles, 1):
            cb_wait_front(cb_in0, 1)
            cb_wait_front(cb_in1, 1)

            tile_regs_acquire()

            add_tiles(cb_in0, cb_in1, 0, 0, 0)

            tile_regs_commit()

            tile_regs_wait()
            pack_tile(dst_reg, cb_out, 0)
            tile_regs_release()

            cb_pop_front(cb_in0, 1)
            cb_pop_front(cb_in1, 1)

            cb_push_back(cb_out, 1)

        return

    @reader_thread()
    def eltwise_binary_read_tiles(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        src0_addr,
        src1_addr,
        num_tiles,
    ):
        tile_size_bytes = get_tile_size(cb_in0)

        tensor_accessor_args = TensorAccessorArgs(2, 0)
        addr_gen0 = TensorAccessor(tensor_accessor_args, src0_addr, tile_size_bytes)
        addr_gen1 = TensorAccessor(tensor_accessor_args, src1_addr, tile_size_bytes)

        for i in range(0, num_tiles, 1):
            cb_reserve_back(cb_in0, 1)
            cb_reserve_back(cb_in1, 1)

            cb_in0_addr = get_write_ptr(cb_in0)
            cb_in1_addr = get_write_ptr(cb_in1)

            noc_async_read_tile(i, addr_gen0, cb_in0_addr)
            noc_async_read_tile(i, addr_gen1, cb_in1_addr)
            noc_async_read_barrier()

            cb_push_back(cb_in0, 1)
            cb_push_back(cb_in1, 1)

        return

    @writer_thread()
    def eltwise_binary_write_tiles(
        cb_out: CircularBuffer,
        dst_addr,
        num_tiles,
    ):
        tile_size_bytes = get_tile_size(cb_out)
        tensor_accessor_args = TensorAccessorArgs(1, 0)
        addr_gen = TensorAccessor(tensor_accessor_args, dst_addr, tile_size_bytes)

        for i in range(0, num_tiles, 1):
            cb_wait_front(cb_out, 1)
            cb_out_addr = get_read_ptr(cb_out)
            noc_async_write_tile(i, addr_gen, cb_out_addr)
            noc_async_write_barrier()
            cb_pop_front(cb_out, 1)

        return

    def invoke(
        self,
        in_tensor0,
        in_tensor1,
        out_tensor,
    ):
        cb_in0 = self.create_cb(in_tensor0, 0)
        cb_in1 = self.create_cb(in_tensor1, 1)
        cb_out = self.create_cb(out_tensor, 2)

        num_tiles = in_tensor0.volume() // 1024
        # print(f"num_tiles: {num_tiles}")

        self.set_tensor_accessor_config([in_tensor0, in_tensor1, out_tensor])

        kernels = [
            self.create_kernel(
                EltwiseBinaryPyKernelOp.eltwise_binary_tiles_add,
                cb_in0,
                cb_in1,
                cb_out,
                num_tiles,
            ),
            self.create_kernel(
                EltwiseBinaryPyKernelOp.eltwise_binary_read_tiles,
                cb_in0,
                cb_in1,
                in_tensor0.buffer_address(),
                in_tensor1.buffer_address(),
                num_tiles,
            ),
            self.create_kernel(
                EltwiseBinaryPyKernelOp.eltwise_binary_write_tiles,
                cb_out,
                out_tensor.buffer_address(),
                num_tiles,
            ),
        ]

        return self.create_program(kernels, [cb_in0, cb_in1, cb_out])


# Device Definitions
device = ttnn.open_device(device_id=0)

# I/O Tensor Definitions
num_tiles = 4
shape = [1, num_tiles, 32, 32]
data0 = torch.rand(shape).to(torch.bfloat16)
data1 = torch.rand(shape).to(torch.bfloat16)

dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

input_tensor0 = ttnn.from_torch(
    data0,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=dram_memory_config,
)

input_tensor1 = ttnn.from_torch(
    data1,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=dram_memory_config,
)

output_tensor = ttnn.allocate_tensor_on_device(
    shape,
    ttnn.bfloat16,
    ttnn.TILE_LAYOUT,
    device,
    dram_memory_config,
)

# Define Custom Generic Op
eltwise_binary_op = EltwiseBinaryPyKernelOp()

# Run tests against the golden "exp" op.
output = eltwise_binary_op(input_tensor0, input_tensor1, output_tensor)
golden = ttnn.add(input_tensor0, input_tensor1)

torch_golden = ttnn.to_torch(golden)
torch_output = ttnn.to_torch(output)
print(f"torch_golden: {torch_golden}")
print(f"torch_output: {torch_output}")

matching = torch.allclose(torch_golden, torch_output)
print(f"Tensors are matching: {matching}")
assert matching
