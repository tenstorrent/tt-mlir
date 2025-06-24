# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

from math import ceil

import ttnn
import torch


class VecAddMulticorePyKernelOp(PyKernelOp):
    def __init__(self, max_core_ranges=None):
        super().__init__()
        self.max_core_ranges = max_core_ranges

    # KERNEL DEFINITIONS
    @compute_thread()
    def add_multicore(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        cb_out: CircularBuffer,
        num_tiles,
        start_tile_id,
    ):
        binary_op_init_common(cb_in0, cb_in1, cb_out)
        add_tiles_init(cb_in0, cb_in1)

        end_tile_id = start_tile_id + num_tiles
        dst_reg = 0

        for i in range(start_tile_id, end_tile_id, 1):
            cb_wait_front(cb_in0, 1)
            cb_wait_front(cb_in1, 1)
            tile_regs_acquire()
            add_tiles(cb_in0, cb_in1, 0, 0, dst_reg)
            tile_regs_commit()

            cb_reserve_back(cb_out, 1)
            tile_regs_wait()
            pack_tile(dst_reg, cb_out, 0)
            tile_regs_release()

            cb_push_back(cb_out, 1)
            cb_pop_front(cb_in0, 1)
            cb_pop_front(cb_in1, 1)
            tile_regs_release()
        return

    @writer_thread()
    def writer_multicore(
        cb_out: CircularBuffer,
        dst_addr,
        num_tiles,
        start_id,
        dst_is_dram: CompiledValue,
    ):
        onetile = 1
        tile_bytes = get_tile_size(cb_out)
        dataformat = get_dataformat(cb_out)

        s0 = get_interleaved_addr_gen_fast(
            dst_is_dram, dst_addr, tile_bytes, dataformat
        )

        end_id = start_id + num_tiles
        for i in range(start_id, end_id, onetile):
            cb_wait_front(cb_out, onetile)
            l1_read_addr = get_read_ptr(cb_out)
            noc_async_write_tile(i, s0, l1_read_addr)
            noc_async_write_barrier()
            cb_pop_front(cb_out, onetile)
        return

    @reader_thread()
    def reader_binary_interleaved(
        cb_in0: CircularBuffer,
        cb_in1: CircularBuffer,
        src_addr0,
        src_addr1,
        num_tiles,
        start_id,
        src0_is_dram: CompiledValue,
        src1_is_dram: CompiledValue,
    ):
        onetile = 1
        tile_bytes0 = get_tile_size(cb_in0)
        dataformat0 = get_dataformat(cb_in0)

        s0 = get_interleaved_addr_gen_fast(
            src0_is_dram, src_addr0, tile_bytes0, dataformat0
        )

        tile_bytes1 = get_tile_size(cb_in1)
        dataformat1 = get_dataformat(cb_in1)

        s1 = get_interleaved_addr_gen_fast(
            src1_is_dram, src_addr1, tile_bytes1, dataformat1
        )

        end_id = start_id + num_tiles
        for i in range(start_id, end_id, onetile):
            cb_reserve_back(cb_in0, onetile)
            cb_reserve_back(cb_in1, onetile)

            src0_write_addr = get_write_ptr(cb_in0)
            src1_write_addr = get_write_ptr(cb_in1)

            noc_async_read_tile(i, s0, src0_write_addr)
            noc_async_read_tile(i, s1, src1_write_addr)

            noc_async_read_barrier()
            cb_push_back(cb_in0, onetile)
            cb_push_back(cb_in1, onetile)
        return

    def define_core_ranges(self, tensors, options):
        core_0 = ttnn.CoreCoord(0, 0)
        if self.max_core_ranges is None:
            core_1 = ttnn.CoreCoord(1, 1)
        else:
            core_1 = self.max_core_ranges
        return ttnn.CoreRangeSet([ttnn.CoreRange(core_0, core_1)])

    def invoke(
        self,  # super() has invoke signature (*tensors, **options)
        a_tensor,
        b_tensor,
        out_tensor,  # Tensor Definitions are positional args
    ):
        cb_in0 = self.create_cb(a_tensor, 0)
        cb_in1 = self.create_cb(b_tensor, 1)
        cb_out = self.create_cb(out_tensor, 2)
        start_id = 0

        is_a_dram = a_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        is_b_dram = b_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
        is_out_dram = out_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM

        num_tiles = ceil(
            max(map(lambda t: t.volume(), [a_tensor, b_tensor, out_tensor])) / 1024
        )

        kernels = [
            self.create_kernel(
                VecAddMulticorePyKernelOp.add_multicore,
                cb_in0,
                cb_in1,
                cb_out,
                num_tiles,
                start_id,
            ),
            self.create_kernel(
                VecAddMulticorePyKernelOp.writer_multicore,
                cb_out,
                out_tensor.buffer_address(),
                num_tiles,
                start_id,
                dst_is_dram=is_out_dram,
            ),
            self.create_kernel(
                VecAddMulticorePyKernelOp.reader_binary_interleaved,
                cb_in0,
                cb_in1,
                a_tensor.buffer_address(),
                b_tensor.buffer_address(),
                num_tiles,
                start_id,
                src0_is_dram=is_a_dram,
                src1_is_dram=is_b_dram,
            ),
        ]

        return self.create_program(kernels, [cb_in0, cb_in1, cb_out])


# Device Definitions
device = ttnn.open_device(device_id=0)

# I/O Tensor Definitions
num_tiles = 4
shape = [1, num_tiles, 32, 32]
data = torch.rand(shape).to(torch.bfloat16)
data2 = torch.rand(shape).to(torch.bfloat16)

dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

a_tensor = ttnn.from_torch(
    data,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=dram_memory_config,
)

b_tensor = ttnn.from_torch(
    data2,
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

# Define Custom Generic Op
core_ranges = None  # Define core ranges here
vecadd_op = VecAddMulticorePyKernelOp()

# Run tests against the golden "exp" op.
output = vecadd_op(a_tensor, b_tensor, output_tensor)
golden = ttnn.add(a_tensor, b_tensor)

torch_golden = ttnn.to_torch(golden)
torch_output = ttnn.to_torch(output)

print(f"a_tensor: {a_tensor}")
print(f"b_tensor: {b_tensor}")
print(f"torch_golden: {torch_golden}")
print(f"torch_output: {torch_output}")

matching = torch.allclose(torch_golden, torch_output)
print(f"Tensors are matching: {matching}")
assert matching
