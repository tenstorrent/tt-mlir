# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from op_definitions import exp


@pytest.mark.skip(
    reason="Error is raised from ttnn first, not JIT frontend.",
)
def test_l1_interleaved_not_supported(device):

    with pytest.raises(ValueError, match="Interleaved L1 tensors are not supported."):
        shape = (32, 32)
        torch_tensor = torch.randn(shape, dtype=torch.float32)

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        )

        ttnn_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        op_jit = ttnn_jit.jit(debug=False)(exp)
        output_tensor = op_jit(ttnn_tensor)


def test_row_major_layout_not_supported(device):

    with pytest.raises(
        ValueError,
        match="Only Layout.Tile tensors are supported. Found layout: Layout.ROW_MAJOR",
    ):

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
            buffer_type=ttnn.BufferType.L1,
            shard_spec=shard_spec,
        )

        ttnn_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        op_jit = ttnn_jit.jit(debug=True)(exp)
        output_tensor = op_jit(ttnn_tensor)
