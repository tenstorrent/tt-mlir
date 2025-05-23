# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

import ttnn
import torch


class EltwiseSFPUPyKernelOp(PyKernelOp):
    # KERNEL DEFINITIONS
    @staticmethod
    @ttkernel_tensix_compile()
    def eltwise_sfpu(cb_in: CircularBuffer, cb_out: CircularBuffer, ct_args=[]):
        per_core_block_cnt = ct_args[0]
        per_core_block_dim = ct_args[1]

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

    def eltwise_sfpu_CT_ARGS(self, tensors, options):
        # Create the CT Args for the eltwise_sfpu kernel
        return [
            options["num_tiles"],  # per_core_block_cnt
            1,  # per_core_block_dim = ct_args[1]
        ]

    def eltwise_sfpu_DEFINES(self, tensors, options):
        return [
            ("SFPU_OP_EXP_INCLUDE", "1"),
            ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"),
        ]

    @staticmethod
    @ttkernel_noc_compile()
    def writer_unary_interleaved(
        cb_in: CircularBuffer, cb_out: CircularBuffer, rt_args, ct_args=[]
    ):
        dst_addr: int = rt_args[0]
        num_tiles = rt_args[1]
        start_id = rt_args[2]

        dst_is_dram = ct_args[0]
        onetile = 1
        tile_bytes = get_tile_size(cb_out)
        dataformat = get_dataformat(cb_out)

        s0 = get_interleaved_addr_gen_fast(
            dst_is_dram > 0, dst_addr, tile_bytes, dataformat
        )

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

    def writer_unary_interleaved_CT_ARGS(self, tensors, options):
        return [
            options["out_cb"],
            options["is_dram_input"],
        ]

    def writer_unary_interleaved_RT_ARGS(self, tensors, options):
        return [
            tensors[1].buffer_address(),
            options["num_tiles"],
            0,  # start_id
        ]

    @staticmethod
    @ttkernel_noc_compile()
    def reader_unary_interleaved(
        cb_in: CircularBuffer, cb_out: CircularBuffer, rt_args, ct_args=[]
    ):
        src_addr: int = rt_args[0]
        num_tiles = rt_args[1]
        start_id = rt_args[2]

        src_is_dram = ct_args[0]  # True
        onetile = 1
        tile_bytes = get_tile_size(cb_in)
        dataformat = get_dataformat(cb_in)

        s0 = get_interleaved_addr_gen_fast(
            src_is_dram > 0, src_addr, tile_bytes, dataformat
        )

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

    def reader_unary_interleaved_CT_ARGS(self, tensors, options):
        return [
            options["is_dram_input"],
        ]

    def reader_unary_interleaved_RT_ARGS(self, tensors, options):
        return [
            tensors[0].buffer_address(),
            options["num_tiles"],
            0,  # start_id
        ]

    # KERNEL SELECTION
    def assign_kernels_and_core_ranges(self, tensors, options):
        # Define the kernel types here to be passed into the config
        core = ttnn.CoreCoord(0, 0)
        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

        return {
            core_ranges: [
                (EltwiseSFPUPyKernelOp.eltwise_sfpu, ttnn.ComputeConfigDescriptor),
                (
                    EltwiseSFPUPyKernelOp.writer_unary_interleaved,
                    ttnn.WriterConfigDescriptor,
                ),
                (
                    EltwiseSFPUPyKernelOp.reader_unary_interleaved,
                    ttnn.ReaderConfigDescriptor,
                ),
            ]
        }

    # Circular Buffers  will be statically created and used for all kernels
    def define_cbs(self, tensors, options):
        # Parse these kernel values from the supplied Options
        input_cb_data_format = options["cb_data_format"]
        cb_total_size = options["cb_total_size"]
        cb_page_size = options["cb_page_size"]
        in_cb = options["in_cb"]
        out_cb = options["out_cb"]

        core_ranges = options["core_ranges"]

        in_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=in_cb,
            data_format=input_cb_data_format,
            page_size=cb_page_size,
        )
        out_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=out_cb,
            data_format=input_cb_data_format,
            page_size=cb_page_size,
        )

        # Set cb formats as artifact to use later on
        self._cb_formats = [in_cb_format, out_cb_format]

        in_cb_descriptor = ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_ranges,
            format_descriptors=[in_cb_format],
        )
        out_cb_descriptor = ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_ranges,
            format_descriptors=[out_cb_format],
        )

        return [in_cb_descriptor, out_cb_descriptor]


# Device Definitions
device = ttnn.open_device(device_id=0)

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
eltwise_exp_op = EltwiseSFPUPyKernelOp(ttnn)

eltwise_exp_op_options = {
    "in_cb": 0,
    "out_cb": 16,
    "is_dram_input": 1,
    "cb_data_format": ttnn.bfloat16,
    "cb_total_size": 2
    * 2
    * 1024,  # tt::DataFormat::Float16_b hard coded to have size 2 * 1024
    "cb_page_size": 2 * 1024,
    "num_tiles": num_tiles,
    "core_ranges": core_ranges,
}

# Run tests against the golden "exp" op.
output = eltwise_exp_op(*io_tensors, **eltwise_exp_op_options)
golden = ttnn.exp(input_tensor)

torch_golden = ttnn.to_torch(golden)
torch_output = ttnn.to_torch(output)

print(f"input_tensor: {input_tensor}")
print(f"torch_golden: {torch_golden}")
print(f"torch_output: {torch_output}")

matching = torch.allclose(torch_golden, torch_output)
print(f"Tensors are matching: {matching}")
assert matching
