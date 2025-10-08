# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel import PyKernelOp, reader_thread, CircularBuffer

import ttnn
import torch
import os


class DPrintPyKernelOp(PyKernelOp):
    @reader_thread()
    def dprint(
        cb_in: CircularBuffer,
        cb_out: CircularBuffer,
        src_addr,
        dst_addr,
        num_tiles,
        start_id,
    ):
        print("Hello world\\n")
        print("cb_in: ", cb_in, "cb_out: ", cb_out, "\\n")
        print(
            "src_addr={}, dst_addr={}, num_tiles={}, start_id={}\\n".format(
                src_addr, dst_addr, num_tiles, start_id
            )
        )

        return

    def invoke(
        self,
        in_tensor,
        out_tensor,
    ):
        cb_in = self.create_cb(in_tensor, 0)
        cb_out = self.create_cb(out_tensor, 1)
        start_id = 0
        num_tiles = 1
        kernels = [
            self.create_kernel(
                DPrintPyKernelOp.dprint,
                cb_in,
                cb_out,
                in_tensor.buffer_address(),
                out_tensor.buffer_address(),
                num_tiles,
                start_id,
            ),
        ]

        return self.create_program(kernels, [])


def main(device):
    shape = [1, 1, 32, 32]
    data0 = torch.zeros(shape).to(torch.bfloat16)

    dummy_tensor = ttnn.from_torch(
        data0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Must pass an input and output tensor to generic
    dprint_op = DPrintPyKernelOp()
    dprint_op(dummy_tensor, dummy_tensor)


if __name__ == "__main__":
    # Set env var to enable dprint
    os.environ["TT_METAL_DPRINT_CORES"] = "0,0"
    device = ttnn.open_device(device_id=0)
    main(device)
    ttnn.close_device(device)
