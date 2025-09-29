# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from math import *
from utils import _get_ttnn_op


def add(a, b):
    return ttnn.add(a, b)


def sub(a, b):
    return ttnn.subtract(a, b)


def mul(a, b):
    return ttnn.multiply(a, b)


def div(a, b):
    return ttnn.divide(a, b)


def logical_and(a, b):
    return ttnn.logical_and(a, b)


def logical_or(a, b):
    return ttnn.logical_or(a, b)


def logical_xor(a, b):
    return ttnn.logical_xor(a, b)


def bitwise_or(a, b):
    return ttnn.bitwise_or(a, b)


def bitwise_and(a, b):
    return ttnn.bitwise_and(a, b)


def bitwise_xor(a, b):
    return ttnn.bitwise_xor(a, b)


def remainder(a, b):
    return ttnn.remainder(a, b)


def pow(a, b):
    return ttnn.pow(a, b)


def atan2(a, b):
    return ttnn.atan2(a, b)


def eq(a, b):
    return ttnn.eq(a, b)


def ne(a, b):
    return ttnn.ne(a, b)


def gt(a, b):
    return ttnn.gt(a, b)


def ge(a, b):
    return ttnn.ge(a, b)


def lt(a, b):
    return ttnn.lt(a, b)


def le(a, b):
    return ttnn.le(a, b)


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
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "op",
    [
        add,
        sub,
        mul,
        div,
        # logical_and, logical_or, logical_xor,
        # bitwise_or, bitwise_and, bitwise_xor, # not a supported FPU op
        # remainder, atan2,# not supported in TTIRToD2M
        pow,
        eq,
        ne,
        gt,
        ge,
        lt,
        le,
    ],
)
def test_binary_ops(device, h, w, max_grid, dtype, op):
    if op == div:
        pytest.xfail("failing allclose for some shapes")
    if op == pow and dtype == torch.float32:
        pytest.xfail("failing allclose for some shapes")
    torch_input_tensor = torch.randn((h, w), dtype=dtype)
    torch_input_tensor_2 = torch.randn((h, w), dtype=dtype)
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

    input_a = ttnn.from_torch(
        torch_input_tensor,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    input_b = ttnn.from_torch(
        torch_input_tensor_2,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(op)
    output_tensor = op_jit(input_a, input_b)
    golden_tensor = golden_op(input_a, input_b)

    print("--------------------------------")
    print("input_a")
    print(input_a)
    print("input_b")
    print(input_b)
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
