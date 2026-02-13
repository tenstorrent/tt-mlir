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
        (grid_h * ELEMENTS_PER_CORE_H, grid_w * ELEMENTS_PER_CORE_W),
        (grid_h, grid_w),
    )
    for grid_h in range(1, 2)
    for grid_w in range(2, 3)
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
    return (-tile_values).to(dtype)


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


def print_mismatch_map(mismatch_flags, name):
    print(f"{name} mismatch map (X=mismatch, .=match):")
    for row in mismatch_flags:
        cells = []
        for idx, is_mismatch in enumerate(row):
            cells.append("X" if is_mismatch else ".")
            if (idx + 1) % TILES_PER_CORE_W == 0 and idx + 1 != len(row):
                cells.append("|")
        print("  " + " ".join(cells))


def assert_tilewise_allclose(
    output_torch,
    golden_torch,
    output_label,
    shape,
    grid,
    input_torch=None,
    tile_size=32,
    atol=0.1,
    rtol=0.1,
):
    height, width = output_torch.shape
    assert height % tile_size == 0 and width % tile_size == 0

    num_tiles_h = height // tile_size
    num_tiles_w = width // tile_size
    mismatched_tiles = []
    mismatch_flags = [[False for _ in range(num_tiles_w)] for _ in range(num_tiles_h)]

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
                mismatch_flags[tile_row][tile_col] = True
                mismatched_tiles.append(
                    (
                        tile_row,
                        tile_col,
                        output_tile,
                        golden_tile,
                        output_tile[0, 0].item(),
                        golden_tile[0, 0].item(),
                    )
                )

    if mismatched_tiles:
        mismatches_per_core = {}
        mismatches_per_offset = {}
        for tile_row, tile_col, *_ in mismatched_tiles:
            core_row = tile_row // TILES_PER_CORE_H
            core_col = tile_col // TILES_PER_CORE_W
            tile_row_in_core = tile_row % TILES_PER_CORE_H
            tile_col_in_core = tile_col % TILES_PER_CORE_W

            core_key = (core_row, core_col)
            offset_key = (tile_row_in_core, tile_col_in_core)
            mismatches_per_core[core_key] = mismatches_per_core.get(core_key, 0) + 1
            mismatches_per_offset[offset_key] = mismatches_per_offset.get(offset_key, 0) + 1

        print(
            f"\n[{output_label}] mismatch summary for shape={shape}, grid={grid} "
            f"(tile grid {num_tiles_h}x{num_tiles_w}):"
        )
        print_mismatch_map(mismatch_flags, name=output_label)
        if input_torch is not None:
            print_tile_grid(input_torch, tile_size=tile_size, name="Input")
        print_tile_grid(output_torch, tile_size=tile_size, name=output_label)
        print_tile_grid(golden_torch, tile_size=tile_size, name="Golden")

        print("Mismatches per core:")
        for (core_row, core_col), count in sorted(mismatches_per_core.items()):
            print(f"  core[{core_row},{core_col}] -> {count}")

        print("Mismatches by tile offset within core:")
        for (tile_row_in_core, tile_col_in_core), count in sorted(
            mismatches_per_offset.items()
        ):
            print(f"  offset[{tile_row_in_core},{tile_col_in_core}] -> {count}")

        print(f"Found {len(mismatched_tiles)} mismatched tiles (showing up to 20):")
        for (
            tile_row,
            tile_col,
            output_tile,
            golden_tile,
            output_val,
            golden_val,
        ) in mismatched_tiles[:20]:
            print(
                f"  tile[{tile_row},{tile_col}] "
                f"output={output_val:g}, golden={golden_val:g}"
            )
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
def test_abs_ttnn_and_jit_match_torch_with_reblock(device, shape, grid):
    input_torch = create_tile_index_tensor(shape, tile_size=TILE_SIZE, dtype=torch.bfloat16)
    golden_torch = torch.abs(input_torch)

    input_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=grid[1], y=grid[0]),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_ttnn = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )

    output_ttnn = abs_op(input_ttnn)
    output_ttnn_torch = output_ttnn.cpu().to_torch()
    assert_tilewise_allclose(
        output_ttnn_torch,
        golden_torch,
        output_label="TTNN",
        shape=shape,
        grid=grid,
        input_torch=input_torch,
    )
    output_ttnn.deallocate()

    jit_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=grid[1], y=grid[0]),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    output_jit = ttnn_jit.jit(
        debug=True, enable_cache=False, memory_config=jit_memory_config
    )(abs_op)(input_ttnn)
    output_jit_torch = output_jit.cpu().to_torch()
    assert_tilewise_allclose(
        output_jit_torch,
        golden_torch,
        output_label="JIT",
        shape=shape,
        grid=grid,
        input_torch=input_torch,
    )
    output_jit.deallocate()
    input_ttnn.deallocate()


@pytest.mark.parametrize(
    "shape, grid",
    GRID_SHAPE_CASES,
    ids=[f"shape_{shape[0]}x{shape[1]}_grid_{grid[0]}x{grid[1]}" for shape, grid in GRID_SHAPE_CASES],
)
def test_abs_ttnn_and_jit_match_torch_no_reblock(device, shape, grid):
    input_torch = create_tile_index_tensor(shape, tile_size=TILE_SIZE, dtype=torch.bfloat16)
    golden_torch = torch.abs(input_torch)

    input_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=grid[1], y=grid[0]),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    input_ttnn = ttnn.from_torch(
        input_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )

    output_ttnn = abs_op(input_ttnn)
    output_ttnn_torch = output_ttnn.cpu().to_torch()
    assert_tilewise_allclose(
        output_ttnn_torch,
        golden_torch,
        output_label="TTNN",
        shape=shape,
        grid=grid,
        input_torch=input_torch,
    )
    output_ttnn.deallocate()

    jit_memory_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.CoreGrid(x=grid[1] * 2, y=grid[0] * 2),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )
    output_jit = ttnn_jit.jit(
        debug=True, enable_cache=False, memory_config=jit_memory_config,
    )(abs_op)(input_ttnn)
    output_jit_torch = output_jit.cpu().to_torch()
    assert_tilewise_allclose(
        output_jit_torch,
        golden_torch,
        output_label="JIT",
        shape=shape,
        grid=grid,
        input_torch=input_torch,
    )
    output_jit.deallocate()
    input_ttnn.deallocate()