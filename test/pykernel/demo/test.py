# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from eltwise_sfpu_demo import EltwiseSFPUPyKernelOp
from vecadd_multicore_demo import VecAddMulticorePyKernelOp


@pytest.mark.usefixtures("device")
def test_eltwise_sfpu(device):
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

    # Define Custom Generic Op
    eltwise_exp_op = EltwiseSFPUPyKernelOp()

    # Run tests against the golden "exp" op.
    output = eltwise_exp_op(input_tensor, output_tensor)
    golden = ttnn.exp(input_tensor)

    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    matching = torch.allclose(torch_golden, torch_output)
    assert matching


@pytest.mark.usefixtures("device")
def test_vecadd_multicore(device):
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
    vecadd_op = VecAddMulticorePyKernelOp()

    # Run tests against the golden "add" op.
    output = vecadd_op(a_tensor, b_tensor, output_tensor)
    golden = ttnn.add(a_tensor, b_tensor)

    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    matching = torch.allclose(torch_golden, torch_output)
    assert matching
