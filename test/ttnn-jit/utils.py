# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
import ttnn
import ttnn_jit
import torch
from typing import Callable, Optional, Dict, Iterable
from ttnn_jit._src.utils import get_maximal_block_sharding_grid


def _get_ttnn_op(func: Callable) -> Optional[Callable]:
    # Return ttnn.<func.__name__> if it exists and is callable
    try:
        attr = getattr(ttnn, func.__name__)
    except AttributeError:
        return None
    return attr if callable(attr) else None


def memory_configs_equal(memory_config1, memory_config2, debug=True):
    equal = (
        memory_config1.memory_layout == memory_config2.memory_layout
        and memory_config1.buffer_type == memory_config2.buffer_type
        and memory_config1.shard_spec == memory_config2.shard_spec
    )
    if debug:
        print("memory_configs_equal", equal)
    return equal


# ----- Input transforms for ops that need constrained inputs -----


def _transform_reciprocal(t: torch.Tensor) -> torch.Tensor:
    """Avoid zeros: abs() then replace exact zeros with 1e-6."""
    t = torch.abs(t)
    return torch.where(t == 0, torch.tensor(1e-6, dtype=t.dtype), t)


def _transform_digamma(t: torch.Tensor) -> torch.Tensor:
    t = t * 1e5
    t = torch.clamp(t, min=1)
    return t


def _transform_sqrt(t: torch.Tensor) -> torch.Tensor:
    """Ensure non-negative values."""
    return torch.abs(t)


def _transform_tan(t: torch.Tensor) -> torch.Tensor:
    """Uniform in [-pi/2 + 0.05, pi/2 - 0.05]."""
    return t.uniform_(-math.pi / 2 + 0.05, math.pi / 2 - 0.05)


def _transform_div(t: torch.Tensor) -> torch.Tensor:
    """Avoid divide-by-zero: abs() then replace values close to zero with 1e-6."""
    t = torch.abs(t)
    return torch.where(t < 1e-3, torch.tensor(1e-6, dtype=t.dtype), t)


# Map op names to their input transforms
_INPUT_TRANSFORMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "reciprocal": _transform_reciprocal,
    "digamma_func": _transform_digamma,
    "sqrt": _transform_sqrt,
    "tan": _transform_tan,
    "div": _transform_div,
}


def _get_input_transform(
    op: Callable,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    """Get the appropriate input transform for an op based on its name."""
    return _INPUT_TRANSFORMS.get(op.__name__)


def get_core_grid_from_device(device):
    """Get the core grid from the device"""
    return (device.core_grid.x - 1, device.core_grid.y - 1)


def get_expected_block_sharded_memory_config(tensor_shape, device):
    """Get expected block sharded memory config for a given tensor shape"""
    max_core_grid = get_core_grid_from_device(device)
    grid = get_maximal_block_sharding_grid(tensor_shape, max_core_grid)

    return ttnn.create_sharded_memory_config(
        shape=tensor_shape,
        core_grid=ttnn.CoreGrid(x=grid[0] + 1, y=grid[1] + 1),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )


def pcc_check(result, golden_result, threshold=0.99):
    """Check PCC between result and golden using ttnn.pearson_correlation_coefficient."""
    result_torch = result.cpu().to_torch()
    golden_torch = golden_result.cpu().to_torch()
    pcc = ttnn.pearson_correlation_coefficient(golden_torch, result_torch)
    print(f"PCC: {pcc} (threshold: {threshold})")
    return pcc >= threshold, pcc


def all_close_check(result, golden_result, atol=1e-1, rtol=1e-1, debug=True):
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
        # Integer tensor: recreate spatial coverage of fp [0,1] in randn and give some overflow headroom
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
    input_transform=None,
):
    torch_tensor = create_torch_tensor(shape, dtype)
    if input_transform is not None:
        torch_tensor = input_transform(torch_tensor)

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


def create_dram_tensor(
    device,
    shape,
    dtype,
    ttnn_dtype=None,
    mesh_mapper=None,
    input_transform=None,
):
    torch_tensor = create_torch_tensor(shape, dtype)
    if input_transform is not None:
        torch_tensor = input_transform(torch_tensor)
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def run_op_test(
    device,
    shape,
    max_grid,
    dtype,
    op,
    num_inputs,
    buffer_type=ttnn.BufferType.L1,
    enable_cache=False,
    shard_strategy=ttnn.ShardStrategy.BLOCK,
    ttnn_dtype=None,
    compile_only=False,
    check_pcc=True,
    check_allclose=False,
    pcc_threshold=0.99,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    memory_config=None,
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
        enable_cache: Whether to enable cache for the JIT-compiled function (default: False)
        ttnn_dtype: Optional ttnn.DataType override (e.g., ttnn.DataType.BFLOAT8_B)
        check_pcc: Whether to check PCC (default: True)
        check_allclose: Whether to check allclose (default: False)
        pcc_threshold: PCC threshold for comparison (default: 0.99)
        math_fidelity: Math fidelity setting for JIT compilation (default: HiFi4)
        memory_config: Optional output memory configuration for the JIT-compiled function
    """
    # Auto-select input transform based on op name
    input_transform = _get_input_transform(op)

    # Create inputs (same tensors used for both JIT and golden)
    if buffer_type == ttnn.BufferType.L1:
        inputs = [
            create_sharded_tile_tensor(
                device,
                shape,
                max_grid,
                dtype,
                shard_strategy,
                ttnn_dtype,
                input_transform=input_transform,
            )
            for _ in range(num_inputs)
        ]
    else:
        inputs = [
            create_dram_tensor(
                device,
                shape,
                dtype,
                ttnn_dtype,
                mesh_mapper=None,
                input_transform=input_transform,
            )
            for _ in range(num_inputs)
        ]

    golden_op = _get_ttnn_op(op)

    op_jit = ttnn_jit.jit(
        compile_only=compile_only,
        debug=True,
        enable_cache=enable_cache,
        math_fidelity=math_fidelity,
        memory_config=memory_config,
    )(op)

    output_tensor = op_jit(*inputs)
    golden_tensor = (golden_op or op)(*inputs)

    print("created inputs:\n", inputs)
    if not compile_only:
        if memory_config is not None:
            assert memory_configs_equal(output_tensor.memory_config(), memory_config)
        else:
            # ttnn-jit will by default set l1 block sharded output memory config
            # get expected memory config to check for
            expected_memory_config = get_expected_block_sharded_memory_config(
                golden_tensor.shape, device
            )
            assert memory_configs_equal(
                output_tensor.memory_config(), expected_memory_config
            )
        print("--------------------------------")
        print("Output:")
        print(output_tensor)
        print("--------------------------------")
        print("Golden:")
        print(golden_tensor)
        print("--------------------------------")
        if check_pcc:
            passed, pcc = pcc_check(output_tensor, golden_tensor, pcc_threshold)
            assert passed, f"PCC check failed: {pcc} < {pcc_threshold}"
        if check_allclose:
            assert all_close_check(output_tensor, golden_tensor)
