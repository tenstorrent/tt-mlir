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


def _build_golden_map(ops: Iterable[Callable]) -> Dict[Callable, Callable]:
    # Build a func ->ttnn op map for provided ops
    result: Dict[Callable, Callable] = {}
    for op in ops:
        ttnn_op = _get_ttnn_op(op)
        if ttnn_op is not None:
            result[op] = ttnn_op
    return result


def memory_configs_equal(memory_config1, memory_config2):
    return (
        memory_config1.memory_layout == memory_config2.memory_layout
        and memory_config1.buffer_type == memory_config2.buffer_type
        and memory_config1.shard_spec == memory_config2.shard_spec
    )


def create_dram_tensor(device, h, w, dtype, int_max=0):
    torch.manual_seed(0)
    if not (dtype.is_floating_point or dtype.is_complex):
        # recreate spatial coverage of fp [0,1] in randn and give some overflow headroom
        high_val = int_max if int_max else torch.iinfo(dtype).max // 2
        torch_tensor = torch.randint(high_val, (h, w), dtype=dtype)
    else:
        if int_max:
            print("Warning: int_max provided for floating point tensor, ignoring.")
        torch_tensor = torch.randn((h, w), dtype=dtype)

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )
    return ttnn.from_torch(
        torch_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def create_sharded_tile_tensor(device, h, w, max_grid, dtype, int_max=0):
    torch.manual_seed(0)
    if not (dtype.is_floating_point or dtype.is_complex):
        # recreate spatial coverage of fp [0,1] in randn and give some overflow headroom
        high_val = int_max if int_max else torch.iinfo(dtype).max // 2
        torch_tensor = torch.randint(high_val, (h, w), dtype=dtype)
    else:
        if int_max:
            print("Warning: int_max provided for floating point tensor, ignoring.")
        torch_tensor = torch.randn((h, w), dtype=dtype)

    start_coord = ttnn.CoreCoord(0, 0)
    end_coord = ttnn.CoreCoord(max_grid[0], max_grid[1])
    core_range = ttnn.CoreRange(start_coord, end_coord)
    core_range_set = ttnn.CoreRangeSet([core_range])

    # TTNN grids are (Width, Height), while tensor shapes are (Height, Width).
    shard_shape_x = h if max_grid[1] == 0 else h // (max_grid[1] + 1)
    shard_shape_y = w if max_grid[0] == 0 else w // (max_grid[0] + 1)

    shard_spec = ttnn.ShardSpec(
        grid=core_range_set,
        shard_shape=[shard_shape_x, shard_shape_y],
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    return ttnn.from_torch(
        torch_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


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


def run_op_test(
    device,
    h,
    w,
    max_grid,
    dtype,
    op,
    num_inputs,
    buffer_type=ttnn.BufferType.L1,
    graph_capture=True,
):
    """
    Common test runner for JIT operations.

    Args:
        device: Device to run the operation on
        h: Height of the input tensor
        w: Width of the input tensor
        max_grid: Maximum grid size for sharded tensors
        dtype: Data type of the input tensor
        op: Operation to test
        num_inputs: Number of input tensors
        buffer_type: Buffer type (L1 or DRAM)
        graph_capture: Whether to use graph capture compiler (default: True)
    """
    if buffer_type == ttnn.BufferType.L1:
        inputs = [
            create_sharded_tile_tensor(device, h, w, max_grid, dtype)
            for _ in range(num_inputs)
        ]
    else:
        inputs = [create_dram_tensor(device, h, w, dtype) for _ in range(num_inputs)]
    print("inputs", inputs)
    golden_op = _get_ttnn_op(op)

    op_jit = ttnn_jit.jit(debug=True, max_grid=max_grid, graph_capture=graph_capture)(
        op
    )
    output_tensor = op_jit(*inputs)
    golden_tensor = (golden_op or op)(*inputs)

    assert memory_configs_equal(
        output_tensor.memory_config(), golden_tensor.memory_config()
    )
    assert all_close_check(output_tensor, golden_tensor)
