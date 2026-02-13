# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn
import ttnn_jit

TILE_SIZE = 32
TILES_PER_CORE_H = 2
TILES_PER_CORE_W = 2
ELEMENTS_PER_CORE_H = TILES_PER_CORE_H * TILE_SIZE
ELEMENTS_PER_CORE_W = TILES_PER_CORE_W * TILE_SIZE

GRID_SHAPE_CASES = [
    (
        (grid_y * ELEMENTS_PER_CORE_H, grid_x * ELEMENTS_PER_CORE_W),
        (grid_x, grid_y),
    )
    for grid_y in range(1, 9)
    for grid_x in range(1, 9)
]


def abs_op(input_tensor):
    return ttnn.abs(input_tensor)


def create_tile_index_tensor(shape, tile_size=32, dtype=torch.bfloat16):
    height, width = shape
    assert height % tile_size == 0 and width % tile_size == 0

    num_tiles_w = width // tile_size
    tile_rows = torch.arange(height, dtype=torch.int32) // tile_size
    tile_cols = torch.arange(width, dtype=torch.int32) // tile_size
    tile_values = tile_rows[:, None] * num_tiles_w + tile_cols[None, :]
    return tile_values.to(dtype)


def print_tile_grid(tensor, tile_size=32, name="Tensor"):
    height, width = tensor.shape
    assert height % tile_size == 0 and width % tile_size == 0
    num_tiles_h = height // tile_size
    num_tiles_w = width // tile_size

    print(f"{name} tile grid ({num_tiles_h}x{num_tiles_w}):")
    for tile_row in range(num_tiles_h):
        row_cells = []
        for tile_col in range(num_tiles_w):
            row_start = tile_row * tile_size
            col_start = tile_col * tile_size
            tile_value = tensor[row_start, col_start].item()
            row_cells.append(f"[{tile_row},{tile_col}]={tile_value:g}")
        print("  " + " | ".join(row_cells))


def assert_tilewise_allclose(output_torch, golden_torch, output_label, tile_size=32, atol=0.1, rtol=0.1):
    height, width = output_torch.shape
    assert height % tile_size == 0 and width % tile_size == 0

    num_tiles_h = height // tile_size
    num_tiles_w = width // tile_size
    mismatched_tiles = []

    for tile_row in range(num_tiles_h):
        for tile_col in range(num_tiles_w):
            row_start = tile_row * tile_size
            row_end = row_start + tile_size
            col_start = tile_col * tile_size
            col_end = col_start + tile_size

            output_tile = output_torch[row_start:row_end, col_start:col_end]
            golden_tile = golden_torch[row_start:row_end, col_start:col_end]
            is_close = torch.allclose(output_tile, golden_tile, atol=atol, rtol=rtol)
            if not is_close:
                mismatched_tiles.append((tile_row, tile_col, output_tile, golden_tile))

    if mismatched_tiles:
        print_tile_grid(output_torch, tile_size=tile_size, name=output_label)
        print_tile_grid(golden_torch, tile_size=tile_size, name="Golden")
        print(f"Found {len(mismatched_tiles)} mismatched tiles:")
        for tile_row, tile_col, output_tile, golden_tile in mismatched_tiles:
            print(f"  Tile [{tile_row}, {tile_col}] mismatch")
            print(f"    {output_label}:")
            print(output_tile)
            print("    Golden:")
            print(golden_tile)

    assert not mismatched_tiles, f"{len(mismatched_tiles)} tile(s) failed allclose in {output_label}"


@pytest.mark.parametrize(
    "shape, grid",
    GRID_SHAPE_CASES,
    ids=[f"shape_{shape[0]}x{shape[1]}_grid_{grid[0]}x{grid[1]}" for shape, grid in GRID_SHAPE_CASES],
)
def test_abs_ttnn_and_jit_match_torch(device, shape, grid):
    input_torch = create_tile_index_tensor(shape, tile_size=TILE_SIZE, dtype=torch.bfloat16)
    golden_torch = torch.abs(input_torch)

    memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=grid[0], y=grid[1]),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_ttnn = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )

    output_ttnn = abs_op(input_ttnn)
    output_ttnn_torch = output_ttnn.cpu().to_torch()
    assert_tilewise_allclose(output_ttnn_torch, golden_torch, output_label="TTNN")
    output_ttnn.deallocate()

    output_jit = ttnn_jit.jit(debug=False, enable_cache=False, memory_config=memory_config)(abs_op)(input_ttnn)
    output_jit_torch = output_jit.cpu().to_torch()
    assert_tilewise_allclose(output_jit_torch, golden_torch, output_label="JIT")
    output_jit.deallocate()
    input_ttnn.deallocate()