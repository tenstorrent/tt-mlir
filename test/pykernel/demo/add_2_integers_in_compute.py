# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

import ttnn
import torch

class Add2IntegersInComputePyKernelOp(PyKernelOp):
    @compute_thread()
    def add_2_integers_in_compute(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
    ):
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

    @reader_thread()
    def add_2_integers_reader_binary_1_tile(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        src0_addr,
        src1_addr,
        src0_bank_id,
        src1_bank_id,
    ):
        src0_noc_addr = get_noc_addr_from_bank_id(src0_bank_id, src0_addr)
        src1_noc_addr = get_noc_addr_from_bank_id(src1_bank_id, src1_addr)

        ublock_size_bytes_0 = get_tile_size(cb_in0)
        ublock_size_bytes_1 = get_tile_size(cb_in1)

        l1_write_addr_in0 = get_write_ptr(cb_in0)
        l1_write_addr_in1 = get_write_ptr(cb_in1)

        cb_reserve_back(cb_in0, 1)
        noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0)
        noc_async_read_barrier()
        cb_push_back(cb_in0, 1)

        cb_reserve_back(cb_in1, 1)
        noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1)
        noc_async_read_barrier()
        cb_push_back(cb_in1, 1)

        return

    @writer_thread()
    def add_2_integers_writer_1_tile(
        cb_out: CircularBuffer,
        dst_addr,
        dst_bank_id,
    ):
        dst_noc_addr = get_noc_addr_from_bank_id(dst_bank_id, dst_addr)

        ublock_size_bytes = get_tile_size(cb_out)
        l1_read_addr = get_read_ptr(cb_out)

        cb_wait_front(cb_out, 1)
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes)
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

        src0_bank_id = 0
        src1_bank_id = 0
        dst_bank_id = 0

        kernels = [
            self.create_kernel(
                Add2IntegersInComputePyKernelOp.add_2_integers_in_compute,
                cb_in0,
                cb_in1,
                cb_out,
            ),
            self.create_kernel(
                Add2IntegersInComputePyKernelOp.add_2_integers_reader_binary_1_tile,
                cb_in0,
                cb_in1,
                in_tensor0.buffer_address(),
                in_tensor1.buffer_address(),
                src0_bank_id,
                src1_bank_id,
            ),
            self.create_kernel(
                Add2IntegersInComputePyKernelOp.add_2_integers_writer_1_tile,
                cb_out,
                out_tensor.buffer_address(),
                dst_bank_id,
            ),
        ]

        return self.create_program(kernels, [cb_in0, cb_in1, cb_out])


# Device Definitions
device = ttnn.open_device(device_id=0)

# I/O Tensor Definitions
shape = [1, 1, 32, 32]
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
add_2_integers_in_compute_op = Add2IntegersInComputePyKernelOp()

# Run tests against the golden "exp" op.
output = add_2_integers_in_compute_op(input_tensor0, input_tensor1, output_tensor)
golden = ttnn.add(input_tensor0, input_tensor1)

torch_golden = ttnn.to_torch(golden)
torch_output = ttnn.to_torch(output)
print(f"torch_golden: {torch_golden}")
print(f"torch_output: {torch_output}")

matching = torch.allclose(torch_golden, torch_output)
print(f"Tensors are matching: {matching}")
assert matching
