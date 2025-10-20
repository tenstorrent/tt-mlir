# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from utils import _get_ttnn_op

COMMON_SHAPE_GRID_PARAMS = [
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
]


def create_dram_tensor(device, h, w, dtype):
    torch.manual_seed(0)
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


def create_sharded_tile_tensor(device, h, w, max_grid, dtype):
    torch.manual_seed(0)
    torch_tensor = torch.randn((h, w), dtype=dtype)

    start_coord = ttnn.CoreCoord(0, 0)
    end_coord = ttnn.CoreCoord(max_grid[0], max_grid[1])
    core_range = ttnn.CoreRange(start_coord, end_coord)
    core_range_set = ttnn.CoreRangeSet([core_range])

    shard_shape_x = h if max_grid[0] == 0 else h // (max_grid[0] + 1)
    shard_shape_y = w if max_grid[1] == 0 else w // (max_grid[1] + 1)

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


def all_close_check(interop_result, golden_result, atol=1e-1, rtol=1e-1):

    print("--------------------------------")
    print("Interop result:")
    print(interop_result)
    print("--------------------------------")
    print("Golden result:")
    print(golden_result)
    print("--------------------------------")

    all_close = torch.allclose(
        interop_result.cpu().to_torch(),
        golden_result.cpu().to_torch(),
        atol=atol,
        rtol=rtol,
    )
    print("all_close", all_close)
    assert all_close


def mul_add(A, B, C):
    matmul_result = ttnn.multiply(B, C)
    add_result = ttnn.add(matmul_result, A)
    return add_result


@pytest.mark.parametrize("h, w, max_grid", COMMON_SHAPE_GRID_PARAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_muladd_jit_l1(device, h, w, max_grid, dtype):
    A = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    B = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    C = create_sharded_tile_tensor(device, h, w, max_grid, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    all_close_check(interop_result, golden_result)


@pytest.mark.parametrize(
    "h, w",
    [
        (32, 32),
        (32, 64),
        (64, 64),
        (64, 128),
        (128, 128),
        (256, 256),
        (256, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_muladd_jit_dram(device, h, w, dtype):
    if (h, w) == (256, 512):
        pytest.xfail("OOM error.")
    max_grid = (0, 0)

    A = create_dram_tensor(device, h, w, dtype)
    B = create_dram_tensor(device, h, w, dtype)
    C = create_dram_tensor(device, h, w, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    all_close_check(interop_result, golden_result)


passing_configs_l1 = {
    torch.bfloat16: [
        (512, 2048),
        (512, 4096),
        (512, 8192),
        (1024, 2048),
        (1024, 4096),
    ],
}


@pytest.mark.parametrize("seq_len", [2048, 4096, 8192, 16384, 32768, 65536])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_large_muladd_nice_seq_len_jit_l1(device, seq_len, hidden_dim, dtype):
    # most test cases fail
    if (hidden_dim, seq_len) not in passing_configs_l1.get(dtype, []):
        pytest.xfail(
            f"Most large L1 configs fail. Skipping ({hidden_dim}, {seq_len}) with dtype {dtype}."
        )

    max_grid = (7, 7)

    A = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)
    B = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)
    C = create_sharded_tile_tensor(device, seq_len, hidden_dim, max_grid, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    all_close_check(interop_result, golden_result, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("seq_len", [2048, 4096, 8192, 16384, 32768, 65536])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_large_muladd_nice_seq_len_jit_dram(device, seq_len, hidden_dim, dtype):
    pytest.xfail("All large DRAM test configurations are failing.")
    max_grid = (0, 0)

    A = create_dram_tensor(device, seq_len, hidden_dim, dtype)
    B = create_dram_tensor(device, seq_len, hidden_dim, dtype)
    C = create_dram_tensor(device, seq_len, hidden_dim, dtype)

    # JIT path
    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    all_close_check(interop_result, golden_result, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("h, w, max_grid", COMMON_SHAPE_GRID_PARAMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_muladd_broadcast_jit_l1(device, h, w, max_grid, dtype):

    # broadcasts require either h or w to be 1
    # but sharding requires at least 32 x 32 (tile size)
    pytest.xfail(
        "Broadcasting requires either h or w to be 1, but sharded tensor must be at least 32 x 32. Assert error."
    )

    A = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    B = create_sharded_tile_tensor(device, h, w, max_grid, dtype)
    # broadcast C
    C = create_sharded_tile_tensor(device, 1, w, max_grid, dtype)

    print("A:", A.cpu().to_torch())
    print("A shape:", A.shape)
    print("B:", B.cpu().to_torch())
    print("B shape:", B.shape)
    print("C:", C.cpu().to_torch())
    print("C shape:", C.shape)

    # JIT path
    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    all_close_check(interop_result, golden_result)


@pytest.mark.parametrize(
    "h, w", [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_muladd_bcast_jit_dram(device, h, w, dtype):
    if (h, w) in [(32, 32)]:
        pytest.xfail("Fails all_close.")
    else:
        pytest.xfail(
            f"Broadcasted shape is incorrectly chosen to be (32, {w}) for all shapes, not ({h}, {w})."
        )
    max_grid = (0, 0)
    A = create_dram_tensor(device, h, w, dtype)
    B = create_dram_tensor(device, h, w, dtype)
    # broadcast C
    C = create_dram_tensor(device, 1, w, dtype)

    print("A:", A.cpu().to_torch())
    print("A shape:", A.shape)
    print("B:", B.cpu().to_torch())
    print("B shape:", B.shape)
    print("C:", C.cpu().to_torch())
    print("C shape:", C.shape)

    # JIT path
    op_jit = ttnn_jit.jit(backend="ttnn", debug=True, max_grid=max_grid)(mul_add)
    interop_result = op_jit(A, B, C)

    # Golden path
    golden_result = mul_add(A, B, C)

    all_close_check(interop_result, golden_result)
