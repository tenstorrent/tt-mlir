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


def exp(input_tensor_a):
    return ttnn.exp(input_tensor_a)


def test_unsupported_tensor_layouts(device):

    # DRAM tensors must be interleaved
    with pytest.raises(ValueError):
        shape = (32, 32)
        torch_tensor = torch.randn(shape, dtype=torch.float32)

        start_coord = ttnn.CoreCoord(0, 0)
        end_coord = ttnn.CoreCoord(0, 0)
        core_range = ttnn.CoreRange(start_coord, end_coord)
        core_range_set = ttnn.CoreRangeSet([core_range])

        shard_spec = ttnn.ShardSpec(
            grid=core_range_set,
            shard_shape=shape,
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.DRAM,
            shard_spec=shard_spec,
        )

        ttnn_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        op_jit = ttnn_jit.jit(debug=True, graph_capture=False)(exp)
        output_tensor = op_jit(ttnn_tensor)
