# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.ast import *
from pykernel.op import PyKernelOp
from pykernel.types import *

import ttnn
import torch
import os


class DPrintPyKernelOp(PyKernelOp):
    @reader_thread()
    def dprint():
        x = 1
        y = 2
        print("Hello world\\n")
        print("Hello", x, "world", y, "goodbye\\n")
        print("Hello {} world Goodbye {} world\\n".format(x, y))

        return

    def invoke(
        self,
        in_tensor,
        out_tensor,
    ):
        kernels = [
            self.create_kernel(
                DPrintPyKernelOp.dprint,
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
