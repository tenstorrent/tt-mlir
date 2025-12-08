# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import (
    _get_ttnn_op,
    all_close_check,
    memory_configs_equal,
    create_dram_tensor,
    create_sharded_tile_tensor,
    run_op_test,
)


def abs(input_tensor):
    return ttnn.abs(input_tensor)


@pytest.mark.parametrize(
    "math_fidelity",
    ["LoFi", "HiFi2", "HiFi3", "HiFi4", "default", "invalid"],
    ids=["LoFi", "HiFi2", "HiFi3", "HiFi4", "default", "invalid"],
)
def test_math_fidelity(device, math_fidelity):
    if math_fidelity == "invalid":
        pytest.xfail("Invalid math fidelity raises runtime error.")

    if math_fidelity == "default":
        run_op_test(
            device,
            (128, 128),
            max_grid=(0, 0),
            dtype=torch.bfloat16,
            op=abs,
            num_inputs=1,
            buffer_type=ttnn.BufferType.L1,
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        return

    run_op_test(
        device,
        (128, 128),
        max_grid=(0, 0),
        dtype=torch.bfloat16,
        op=abs,
        num_inputs=1,
        buffer_type=ttnn.BufferType.L1,
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        math_fidelity=math_fidelity,
    )
