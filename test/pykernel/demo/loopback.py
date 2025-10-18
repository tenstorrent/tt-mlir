# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

import ttnn
import torch

class LoopbackPyKernelOp(PyKernelOp):
    @reader_thread()
    def loopback_dram_copy(
        cb_in_out: CircularBuffer,
        dram_buffer_src_addr,
        dram_buffer_dst_addr,
        num_tiles,
    ):
        tile_size_bytes = get_tile_size(cb_in_out)
        tensor_accessor_args = TensorAccessorArgs(1, 0)
        addr_gen_src = TensorAccessor(tensor_accessor_args, dram_buffer_src_addr, tile_size_bytes)
        addr_gen_dst = TensorAccessor(tensor_accessor_args, dram_buffer_dst_addr, tile_size_bytes)
        
        for i in range(0, num_tiles, 1):
            cb_reserve_back(cb_in_out, 1)
            cb_in_addr = get_write_ptr(cb_in_out)
            noc_async_read_tile(i, addr_gen_src, cb_in_addr)
            noc_async_read_barrier()

            noc_async_write_tile(i, addr_gen_dst, cb_in_addr)
            noc_async_write_barrier()

        return

    def invoke(
        self,
        in_tensor0,
        out_tensor,
    ):
        cb_in_out = self.create_cb(in_tensor0, 0)
        num_tiles = in_tensor0.volume() // 1024

        self.set_tensor_accessor_config([in_tensor0, out_tensor])

        kernels = [
            self.create_kernel(
                LoopbackPyKernelOp.loopback_dram_copy,
                cb_in_out,
                in_tensor0.buffer_address(),
                out_tensor.buffer_address(),
                num_tiles,
            ),
        ]

        return self.create_program(kernels, [cb_in_out])


# Device Definitions
device = ttnn.open_device(device_id=0)

# # I/O Tensor Definitions
num_tiles = 4
shape = [1, num_tiles, 32, 32]
data0 = torch.rand(shape).to(torch.bfloat16)

dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

input_tensor0 = ttnn.from_torch(
    data0,
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
loopback_op = LoopbackPyKernelOp()

output = loopback_op(input_tensor0, output_tensor)
golden = input_tensor0

torch_golden = ttnn.to_torch(golden)
torch_output = ttnn.to_torch(output)
print(f"torch_golden: {torch_golden}")
print(f"torch_output: {torch_output}")

matching = torch.allclose(torch_golden, torch_output)
print(f"Tensors are matching: {matching}")
assert matching
