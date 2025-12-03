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


@pytest.mark.parametrize("use_graph_capture", [True, False])
def test_l1_interleaved_not_supported(device, use_graph_capture):

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

        op_jit = ttnn_jit.jit(debug=True, graph_capture=use_graph_capture)(exp)
        output_tensor = op_jit(ttnn_tensor)


@pytest.mark.parametrize("use_graph_capture", [True, False])
def test_nd_sharded_not_supported(device, use_graph_capture):

    with pytest.raises(
        ValueError,
        match="Tensor is sharded but no legacy shard spec is present. ND Sharded tensors are not supported yet.",
    ):
        shape = (4, 512, 768)
        core_ranges = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))}
        )

        nd_spec_batch_seq = ttnn.TensorSpec(
            shape=shape,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            buffer_type=ttnn.BufferType.L1,
        ).sharded_across_dims([0, 1], core_ranges)

        torch_tensor = torch.randn(tuple(nd_spec_batch_seq.shape))
        batch_seq_sharded = ttnn.from_torch(
            torch_tensor, spec=nd_spec_batch_seq, device=device
        )

        op_jit = ttnn_jit.jit(debug=True, graph_capture=use_graph_capture)(exp)
        output_tensor = op_jit(batch_seq_sharded)


@pytest.mark.parametrize("use_graph_capture", [True, False])
def test_row_major_layout_not_supported(device, use_graph_capture):

    with pytest.raises(
        ValueError,
        match="Only Layout.TILE tensors are supported. Found layout: Layout.ROW_MAJOR",
    ):
        shape = (32, 32)
        torch_tensor = torch.randn(shape, dtype=torch.float32)

        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        )

        ttnn_tensor = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        op_jit = ttnn_jit.jit(debug=True, graph_capture=use_graph_capture)(exp)
        output_tensor = op_jit(ttnn_tensor)
