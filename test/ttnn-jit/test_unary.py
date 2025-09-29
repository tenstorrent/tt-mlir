# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from math import *
from utils import _get_ttnn_op


def run_unary_op(device, h, w, max_grid, dtype, op):
    # Note: both torch.float16 and torch.bfloat16 will convert to ttnn.bfloat16
    torch_input_tensor = torch.randn((h, w), dtype=dtype)
    golden_op = _get_ttnn_op(op)

    start_coord = ttnn.CoreCoord(0, 0)
    end_coord = ttnn.CoreCoord(max_grid[0], max_grid[1])
    core_range = ttnn.CoreRange(start_coord, end_coord)
    core_range_set = ttnn.CoreRangeSet([core_range])

    shard_shape_x = h if max_grid[0] == 0 else h // (max_grid[0] + 1)
    shard_shape_y = w if max_grid[1] == 0 else w // (max_grid[1] + 1)

    shard_spec = ttnn.ShardSpec(
        grid=core_range_set,
        shard_shape=[
            shard_shape_x,
            shard_shape_y,
        ],
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_mode=ttnn.ShardMode.PHYSICAL,
    )

    memory_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    print("torch_input_tensor: ", torch_input_tensor)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    print("input_ttnn_tensor: ", input_tensor)
    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(op)
    output_tensor = op_jit(input_tensor)
    print("input tensor: ", input_tensor)
    golden_tensor = golden_op(input_tensor)

    print("--------------------------------")
    print("input_tensor")
    print(input_tensor)
    print("--------------------------------")
    print("output_tensor")
    print(output_tensor)
    print("--------------------------------")
    print("golden_tensor")
    print(golden_tensor)
    print("--------------------------------")

    all_close = torch.allclose(
        output_tensor.cpu().to_torch(),
        golden_tensor.cpu().to_torch(),
        atol=1e-1,
        rtol=1e-1,
    )
    print("all_close", all_close)
    assert all_close


def abs(input_tensor):
    return ttnn.abs(input_tensor)


def exp(input_tensor):
    return ttnn.exp(input_tensor)


def log(input_tensor):
    return ttnn.log(input_tensor)


def cos(input_tensor):
    return ttnn.cos(input_tensor)


def sin(input_tensor):
    return ttnn.sin(input_tensor)


def tan(input_tensor):
    return ttnn.tan(input_tensor)


def cbrt(input_tensor):
    return ttnn.cbrt(input_tensor)


def ceil(input_tensor):
    return ttnn.ceil(input_tensor)


def sign(input_tensor):
    return ttnn.sign(input_tensor)


def erf(input_tensor):
    return ttnn.erf(input_tensor)


def erfc(input_tensor):
    return ttnn.erfc(input_tensor)


def floor(input_tensor):
    return ttnn.floor(input_tensor)


def gelu(input_tensor):
    return ttnn.gelu(input_tensor)


def logical_not(input_tensor):
    return ttnn.logical_not(input_tensor)


def bitwise_not(input_tensor):
    return ttnn.bitwise_not(input_tensor)


def reciprocal(input_tensor):
    return ttnn.reciprocal(input_tensor)


def sqrt(input_tensor):
    return ttnn.sqrt(input_tensor)


def rsqrt(input_tensor):
    return ttnn.rsqrt(input_tensor)


# sweep shapes, grid size, dtype
@pytest.mark.parametrize(
    "h , w, max_grid",
    [
        (32, 32, (0, 0)),
        (32, 64, (0, 0)),
        (64, 64, (0, 0)),
        (64, 128, (0, 0)),
        (128, 128, (0, 0)),
        (256, 256, (7, 7)),
        (256, 512, (7, 7)),
        (512, 512, (7, 7)),
        (512, 1024, (7, 7)),
        (1024, 1024, (7, 7)),
        (1024, 2048, (7, 7)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "op",
    [
        abs,
        exp,
        log,
        cos,
        sin,
        ceil,
        floor,
        gelu,
        logical_not,
        reciprocal,
        rsqrt,
        # cbrt, sign, erf, erfc, bitwise_not # <- not supported in TTIRToD2M
        # tan, sqrt # <- always fails allclose
    ],
)
def test_unary_op(device, h, w, max_grid, dtype, op):
    if op in [log, ceil, floor, rsqrt] and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes for float32")
    if op == abs and max_grid == (0, 0) and dtype == torch.bfloat16:
        pytest.skip("already tested in test_unary_op_single_core_sweep")

    run_unary_op(device, h, w, max_grid, dtype, op)


# Note: anything bigger than 256x256 will fail allclose for bfloat16
# looks like data corruption in the runtime..?
@pytest.mark.parametrize(
    "h, w", [(h, w) for h in range(32, 256, 32) for w in range(32, 256, 32)]
)
def test_unary_op_single_core_sweep(device, h, w):
    run_unary_op(device, h, w, (0, 0), torch.bfloat16, abs)
