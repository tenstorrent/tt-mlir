# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import ttnn_jit
import torch
from typing import Callable, Optional, Dict, Iterable


def _get_ttnn_op(func: Callable) -> Optional[Callable]:
    # Return ttnn.<func.__name__> if it exists and is callable
    try:
        attr = getattr(ttnn, func.__name__)
    except AttributeError:
        return None
    return attr if callable(attr) else None


def memory_configs_equal(memory_config1, memory_config2):
    return (
        memory_config1.memory_layout == memory_config2.memory_layout
        and memory_config1.buffer_type == memory_config2.buffer_type
        and memory_config1.shard_spec == memory_config2.shard_spec
    )


def get_block_sharding_grid(shape):
    """Infer a TTNN grid/end coord for block sharding the given logical tensor shape"""
    assert len(shape) == 2, f"Only 2D shapes are supported"
    tile_shape = [shape[0] // 32, shape[1] // 32]
    grid = []
    for dim in tile_shape:
        for grid_dim in reversed(range(8)):
            if dim % (grid_dim + 1) == 0:
                grid.append(grid_dim)
                break
    return list(reversed(grid))


def all_close_check(result, golden_result, atol=1e-1, rtol=1e-1, debug=True):
    if debug:
        print("--------------------------------")
        print("Result:")
        print(result)
        print("--------------------------------")
        print("Golden result:")
        print(golden_result)
        print("--------------------------------")

    all_close = torch.allclose(
        result.cpu().to_torch(),
        golden_result.cpu().to_torch(),
        atol=atol,
        rtol=rtol,
    )

    if debug:
        print("all_close", all_close)

    return all_close


def create_torch_tensor(shape, dtype):
    if not (dtype.is_floating_point or dtype.is_complex):
        # recreate spatial coverage of fp [0,1] in randn and give some overflow headroom
        high_val = torch.iinfo(dtype).max // 2
        torch_tensor = torch.randint(high_val, shape, dtype=dtype)
    else:
        torch_tensor = torch.randn(shape, dtype=dtype)
    return torch_tensor


def create_sharded_tile_tensor(
    device,
    shape,
    max_grid,
    dtype,
    shard_strategy=ttnn.ShardStrategy.BLOCK,
    ttnn_dtype=None,
):
    torch_tensor = create_torch_tensor(shape, dtype)
    # IMPORTANT: TTNN grids are (Width, Height), while tensor shapes are (Height, Width).
    # We add 1 to the grid dimensions to account for the zero-based indexing.
    memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=max_grid[0] + 1, y=max_grid[1] + 1),
        strategy=shard_strategy,
        use_height_and_width_as_shard_shape=False,
    )
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def create_dram_tensor(device, shape, dtype, ttnn_dtype=None):
    torch_tensor = create_torch_tensor(shape, dtype)
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def run_op_test(
    device,
    shape,
    max_grid,
    dtype,
    op,
    num_inputs,
    buffer_type=ttnn.BufferType.L1,
    graph_capture=True,
    enable_cache=False,
    shard_strategy=ttnn.ShardStrategy.BLOCK,
    ttnn_dtype=None,
    compile_only=False,
):
    """
    Common test runner for JIT operations.

    Args:
        device: Device to run the operation on
        shape: Shape of the input tensor
        max_grid: Maximum grid size for sharded tensors
        dtype: Torch dtype of the input tensor
        op: Operation to test
        num_inputs: Number of input tensors
        buffer_type: Buffer type (L1 or DRAM)
        graph_capture: Whether to use graph capture compiler (default: True)
        enable_cache: Whether to enable cache for the JIT-compiled function (default: False)
        ttnn_dtype: Optional ttnn.DataType override (e.g., ttnn.DataType.BFLOAT8_B)
    """
    if buffer_type == ttnn.BufferType.L1:
        inputs = [
            create_sharded_tile_tensor(
                device, shape, max_grid, dtype, shard_strategy, ttnn_dtype
            )
            for _ in range(num_inputs)
        ]
    else:
        inputs = [
            create_dram_tensor(device, shape, dtype, ttnn_dtype)
            for _ in range(num_inputs)
        ]
    golden_op = _get_ttnn_op(op)

    op_jit = ttnn_jit.jit(
        compile_only=compile_only,
        debug=True,
        enable_cache=enable_cache,
        graph_capture=graph_capture,
    )(op)
    output_tensor = op_jit(*inputs)
    golden_tensor = (golden_op or op)(*inputs)

    print("created inputs:\n", inputs)
    if not compile_only:
        assert memory_configs_equal(
            output_tensor.memory_config(), golden_tensor.memory_config()
        )
        assert all_close_check(output_tensor, golden_tensor)
