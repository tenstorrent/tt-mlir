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
)

def create_rank_n_sharded_tile_tensor(device, shape, max_grid, dtype, int_max=0):
    torch.manual_seed(0)
    if 





def abs(input_tensor):
    return ttnn.abs(input_tensor)

#rank 3 tensor test
@pytest.mark.parametrize("rank_1, rank_2, rank_3", [(4, 5, 6), (2, 3, 4)])
@pytest.mark.parametrize("max_grid", (7, 7))
@pytest.mark.parametrize("op", [abs])
def test_rank_3(device, rank_1, rank_2, rank_3, max_grid, op):
    input_tensor = create_rank_n_sharded_tile_tensor(
        device,
        (rank_1, rank_2, rank_3),
        max_grid,
        dtype=torch.float32,
    )

    # JIT compile the operation
    jit_op = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(op)
    output_tensor = op_jit(input_tensor)

    golden_tensor = op(input_tensor)

    assert memory_configs_equal(output_tensor.memory_config(), golden_tensor.memory_config())
    assert all_close_check(output_tensor, golden_tensor)
