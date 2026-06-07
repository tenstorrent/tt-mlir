# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Float tile reductions exposed through the d2m-jit block DSL.

Coverage mirrors `test_matmul.py` but is broader because reductions are
expected to produce defined values without an explicit accumulator pre-fill:

1. `test_reduce_sum_compiles_and_runs` -- baseline shape/dtype smoke test.
2. Parameterized correctness over several layout shapes, block shapes, and
   execution grids for row and column reduction directions.
3. bf16 coverage for reduction scaler type selection.
4. Invalid dim rejection for the Python DSL surface.
"""

import pytest
import torch
import d2m_jit as d2m


@d2m.kernel
def k_reduce_sum_cols(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            reduced = reduce_sum(x, 1)
            remote_store(out_t, [m_off + m, 0], reduced)


@d2m.kernel
def k_reduce_max_rows(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [0, n_off + n], x.reduce_max(0))


@d2m.kernel
def k_reduce_mean_cols_negative_dim(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, 0], reduce_mean(x, -1))


@d2m.kernel
def k_reduce_sum_rows_negative_dim(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [0, n_off + n], x.reduce_sum(-2))


@d2m.kernel
def k_center_cols_with_implicit_bcast(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], x - reduce_mean(x, 1))


@d2m.kernel
def k_reduce_sum_cols_out(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, 0], reduce_sum(x, 1))


@d2m.kernel
def k_reduce_max_rows_out(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [0, n_off + n], x.reduce_max(0))


@d2m.kernel
def k_reduce_mean_cols_out(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, 0], x.reduce_mean(1))


@d2m.kernel
def k_reduce_sum_cols_tile(in_t, out_t, m_blocks, n_tile):
    m_off = core_index(0) * m_blocks
    for m in range(m_blocks):
        x = remote_load(in_t, [m_off + m, n_tile])
        remote_store(out_t, [m_off + m, 0], reduce_sum(x, 1))


@d2m.kernel
def k_reduce_sum_cols_accumulate_tile(in_t, acc_t, out_t, m_blocks, n_tile):
    m_off = core_index(0) * m_blocks
    for m in range(m_blocks):
        x = remote_load(in_t, [m_off + m, n_tile])
        acc = remote_load(acc_t, [m_off + m, 0])
        remote_store(out_t, [m_off + m, 0], acc + reduce_sum(x, 1))


@d2m.kernel
def k_reduce_max_rows_tile(in_t, out_t, m_tile, n_blocks):
    n_off = core_index(1) * n_blocks
    for n in range(n_blocks):
        x = remote_load(in_t, [m_tile, n_off + n])
        remote_store(out_t, [0, n_off + n], x.reduce_max(0))


@d2m.kernel
def k_reduce_max_rows_accumulate_tile(in_t, acc_t, out_t, m_tile, n_blocks):
    n_off = core_index(1) * n_blocks
    for n in range(n_blocks):
        x = remote_load(in_t, [m_tile, n_off + n])
        acc = remote_load(acc_t, [0, n_off + n])
        remote_store(out_t, [0, n_off + n], acc.maximum(x.reduce_max(0)))


_LAYOUT_CASES = [
    pytest.param((32, 32), (1, 1), (1, 1), id="single-tile-grid-1x1"),
    pytest.param((64, 64), (1, 1), (2, 2), id="one-tile-per-core-grid-2x2"),
    pytest.param((64, 64), (2, 1), (1, 1), id="two-row-tiles-per-block"),
    pytest.param((64, 64), (1, 2), (1, 1), id="two-col-tiles-per-block"),
    pytest.param((96, 64), (1, 1), (3, 2), id="rectangular-grid-3x2"),
]

_COL_REDUCTION_CASES = [
    pytest.param((32, 32), (1, 1), (1, 1), id="single-tile-grid-1x1"),
    pytest.param((64, 32), (1, 1), (2, 1), id="cols-fit-on-core-grid-2x1"),
    pytest.param((64, 64), (1, 2), (1, 1), id="two-col-tiles-per-block"),
]

_ROW_REDUCTION_CASES = [
    pytest.param((32, 32), (1, 1), (1, 1), id="single-tile-grid-1x1"),
    pytest.param((32, 64), (1, 1), (1, 2), id="rows-fit-on-core-grid-1x2"),
    pytest.param((64, 64), (2, 1), (1, 1), id="two-row-tiles-per-block"),
]

_IMPLICIT_BCAST_CASES = [
    pytest.param((32, 32), (1, 1), (1, 1), id="single-tile-grid-1x1"),
    pytest.param((64, 64), (1, 1), (2, 2), id="one-tile-per-core-grid-2x2"),
    pytest.param((64, 64), (2, 1), (1, 1), id="two-row-tiles-per-block"),
    pytest.param((96, 64), (1, 1), (3, 2), id="rectangular-grid-3x2"),
]


def _make_layout(
    shape=(32, 32),
    dtype=d2m.float32,
    block_shape=(1, 1),
    grid_shape=(1, 1),
):
    return d2m.Layout(
        shape=shape,
        dtype=dtype,
        block_shape=list(block_shape),
        grid_shape=list(grid_shape),
    )


def _blocks_per_core(layout, grid):
    m_tiles = layout.logical_shape[0] // 32
    n_tiles = layout.logical_shape[1] // 32
    assert m_tiles % (layout.block_shape[0] * grid[0]) == 0
    assert n_tiles % (layout.block_shape[1] * grid[1]) == 0
    return (
        m_tiles // layout.block_shape[0] // grid[0],
        n_tiles // layout.block_shape[1] // grid[1],
    )


def _run(kernel, tensor, layout=None, grid=(1, 1)):
    layout = layout or _make_layout()
    out = d2m.empty(layout)
    m_blocks, n_blocks = _blocks_per_core(layout, grid)
    kernel(d2m.to_layout(tensor, layout), out, m_blocks, n_blocks, grid=grid)
    return out.to_host()


def _run_reduced(kernel, tensor, dim, layout=None, grid=(1, 1)):
    layout = layout or _make_layout()
    output_layout = d2m.reduction_layout(layout, dim)
    out = d2m.empty(output_layout)
    m_blocks, n_blocks = _blocks_per_core(layout, grid)
    kernel(d2m.to_layout(tensor, layout), out, m_blocks, n_blocks, grid=grid)
    return out.to_host()


def _run_with_output_layout(kernel, tensor, input_layout, output_layout, grid):
    out = d2m.empty(output_layout)
    m_blocks, n_blocks = _blocks_per_core(input_layout, grid)
    kernel(d2m.to_layout(tensor, input_layout), out, m_blocks, n_blocks, grid=grid)
    return out.to_host()


def _tile_sum_input(shape, dtype=torch.float32):
    row_values = torch.arange(shape[0], dtype=torch.float32).reshape(shape[0], 1)
    tensor = row_values.repeat(1, shape[1]) / 100.0
    return tensor.to(dtype)


def _tile_max_input(shape, dtype=torch.float32):
    row_bias = torch.linspace(-0.5, 0.5, shape[0], dtype=torch.float32).reshape(
        shape[0], 1
    )
    col_values = torch.linspace(-1.0, 1.0, shape[1], dtype=torch.float32).reshape(
        1, shape[1]
    )
    return (row_bias + col_values).to(dtype)


def _assert_reduce_sum_cols(result, tensor, atol=0.05):
    expected = tensor.sum(dim=1, keepdim=True)
    diff = (expected.to(torch.float32) - result.to(torch.float32)).abs().max().item()
    assert diff < atol, f"reduce_sum cols: max diff {diff}"


def _assert_reduce_mean_cols(result, tensor, atol=0.05):
    expected = tensor.mean(dim=1, keepdim=True)
    diff = (expected.to(torch.float32) - result.to(torch.float32)).abs().max().item()
    assert diff < atol, f"reduce_mean cols: max diff {diff}"


def _assert_reduce_max_rows(result, tensor, atol=0.05):
    expected = tensor.max(dim=0, keepdim=True).values
    diff = (expected.to(torch.float32) - result.to(torch.float32)).abs().max().item()
    assert diff < atol, f"reduce_max rows: max diff {diff}"


def test_reduce_sum_compiles_and_runs():
    tensor = _tile_sum_input((32, 32))
    layout = _make_layout()
    result = _run_reduced(k_reduce_sum_cols, tensor, 1, layout=layout)

    assert tuple(result.shape) == (32, 1)
    assert result.dtype == torch.float32


@pytest.mark.parametrize("shape,block_shape,grid", _COL_REDUCTION_CASES)
def test_reduce_sum_cols_correctness(shape, block_shape, grid):
    """Free-function form: `reduce_sum(x, 1)` reduces each tile's columns."""
    layout = _make_layout(shape=shape, block_shape=block_shape, grid_shape=grid)
    tensor = _tile_sum_input(shape)
    result = _run_reduced(k_reduce_sum_cols, tensor, 1, layout=layout, grid=grid)

    _assert_reduce_sum_cols(result, tensor, atol=0.1)


@pytest.mark.parametrize("shape,block_shape,grid", _ROW_REDUCTION_CASES)
def test_reduce_max_rows_method_form_correctness(shape, block_shape, grid):
    """Method form: `x.reduce_max(0)` reduces each tile's rows."""
    layout = _make_layout(shape=shape, block_shape=block_shape, grid_shape=grid)
    tensor = _tile_max_input(shape)
    result = _run_reduced(k_reduce_max_rows, tensor, 0, layout=layout, grid=grid)

    _assert_reduce_max_rows(result, tensor)


@pytest.mark.parametrize("shape,block_shape,grid", _COL_REDUCTION_CASES)
def test_reduce_mean_cols_negative_dim_correctness(shape, block_shape, grid):
    """Negative dim form: `reduce_mean(x, -1)` aliases column reduction."""
    layout = _make_layout(shape=shape, block_shape=block_shape, grid_shape=grid)
    tensor = _tile_sum_input(shape)
    result = _run_reduced(
        k_reduce_mean_cols_negative_dim, tensor, 1, layout=layout, grid=grid
    )

    _assert_reduce_mean_cols(result, tensor)


@pytest.mark.parametrize("shape,block_shape,grid", _ROW_REDUCTION_CASES)
def test_reduce_sum_rows_negative_dim_correctness(shape, block_shape, grid):
    """Method + negative dim form: `x.reduce_sum(-2)` aliases row reduction."""
    layout = _make_layout(shape=shape, block_shape=block_shape, grid_shape=grid)
    tensor = _tile_max_input(shape)
    result = _run_reduced(
        k_reduce_sum_rows_negative_dim, tensor, 0, layout=layout, grid=grid
    )

    expected = tensor.sum(dim=0, keepdim=True)
    diff = (expected - result).abs().max().item()
    assert diff < 0.1, f"reduce_sum rows: max diff {diff}"


def test_reduce_sum_bf16_dtype():
    bf16_layout = _make_layout(shape=(32, 32), dtype=d2m.bfloat16)
    tensor = _tile_sum_input((32, 32), dtype=torch.bfloat16)

    result = _run_reduced(k_reduce_sum_cols, tensor, 1, layout=bf16_layout)

    assert result.dtype == torch.bfloat16
    _assert_reduce_sum_cols(result, tensor, atol=0.15)


@pytest.mark.parametrize("shape,block_shape,grid", _IMPLICIT_BCAST_CASES)
def test_reduce_mean_cols_implicit_broadcast(shape, block_shape, grid):
    layout = _make_layout(shape=shape, block_shape=block_shape, grid_shape=grid)
    tensor = _tile_sum_input(shape)
    result = _run(k_center_cols_with_implicit_bcast, tensor, layout=layout, grid=grid)

    expected = torch.empty_like(tensor)
    block_width = block_shape[1] * 32
    for col_start in range(0, tensor.shape[1], block_width):
        block = tensor[:, col_start : col_start + block_width]
        expected[:, col_start : col_start + block_width] = block - block.mean(
            dim=1, keepdim=True
        )
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"implicit reduce broadcast: max diff {diff}"


def test_reduce_sum_cols_output_layout():
    input_layout = _make_layout(shape=(64, 32), grid_shape=(2, 1))
    output_layout = d2m.reduction_layout(input_layout, 1)
    tensor = _tile_sum_input((64, 32))
    result = _run_with_output_layout(
        k_reduce_sum_cols_out,
        tensor,
        input_layout,
        output_layout,
        grid=(2, 1),
    )

    assert tuple(result.shape) == (64, 1)
    expected = tensor.sum(dim=1, keepdim=True)
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"reduce_sum(dim=1): max diff {diff}"


def test_reduce_max_rows_output_layout():
    input_layout = _make_layout(shape=(32, 64), grid_shape=(1, 2))
    output_layout = d2m.reduction_layout(input_layout, 0)
    tensor = _tile_max_input((32, 64))
    result = _run_with_output_layout(
        k_reduce_max_rows_out,
        tensor,
        input_layout,
        output_layout,
        grid=(1, 2),
    )

    assert tuple(result.shape) == (1, 64)
    expected = tensor.max(dim=0, keepdim=True).values
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"reduce_max(dim=0): max diff {diff}"


def test_reduce_sum_cols_multi_tile_single_core():
    input_layout = _make_layout(shape=(32, 64), block_shape=(1, 2), grid_shape=(1, 1))
    output_layout = d2m.reduction_layout(input_layout, 1)
    tensor = _tile_sum_input((32, 64))
    result = _run_with_output_layout(
        k_reduce_sum_cols_out,
        tensor,
        input_layout,
        output_layout,
        grid=(1, 1),
    )

    assert tuple(result.shape) == (32, 1)
    expected = tensor.sum(dim=1, keepdim=True)
    diff = (expected - result).abs().max().item()
    assert diff < 0.1, f"single-core multi-tile reduce_sum(dim=1): max diff {diff}"


def test_reduce_max_rows_multi_tile_single_core():
    input_layout = _make_layout(shape=(64, 32), block_shape=(2, 1), grid_shape=(1, 1))
    output_layout = d2m.reduction_layout(input_layout, 0)
    tensor = _tile_max_input((64, 32))
    result = _run_with_output_layout(
        k_reduce_max_rows_out,
        tensor,
        input_layout,
        output_layout,
        grid=(1, 1),
    )

    assert tuple(result.shape) == (1, 32)
    expected = tensor.max(dim=0, keepdim=True).values
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"single-core multi-tile reduce_max(dim=0): max diff {diff}"


def test_reduce_mean_cols_multi_tile_single_core():
    input_layout = _make_layout(shape=(32, 64), block_shape=(1, 2), grid_shape=(1, 1))
    output_layout = d2m.reduction_layout(input_layout, 1)
    tensor = _tile_sum_input((32, 64))
    result = _run_with_output_layout(
        k_reduce_mean_cols_out,
        tensor,
        input_layout,
        output_layout,
        grid=(1, 1),
    )

    assert tuple(result.shape) == (32, 1)
    expected = tensor.mean(dim=1, keepdim=True)
    diff = (expected - result).abs().max().item()
    assert diff < 0.1, f"single-core multi-tile reduce_mean(dim=1): max diff {diff}"


def test_reduction_layout_rejects_cross_core_reduction():
    layout = _make_layout(shape=(64, 64), grid_shape=(2, 2))
    with pytest.raises(ValueError, match="fits on one core"):
        d2m.reduction_layout(layout, 1)


def test_reduction_layout_allows_multiple_blocks_on_one_core():
    layout = _make_layout(shape=(64, 64), block_shape=(1, 1), grid_shape=(1, 1))
    output_layout = d2m.reduction_layout(layout, 1)

    assert output_layout.logical_shape == [64, 1]
    assert output_layout.block_shape == [1, 1]
    assert output_layout.grid_shape == [1, 1]


def test_reduce_sum_cols_cross_tile_output_layout():
    input_layout = _make_layout(shape=(64, 64), grid_shape=(2, 2))
    output_layout = d2m.reduction_layout(input_layout, 1, allow_cross_tile=True)
    tensor = _tile_sum_input((64, 64))
    m_blocks = input_layout.logical_shape[0] // 32 // 2
    n_tiles = input_layout.logical_shape[1] // 32

    out = d2m.empty(output_layout)
    k_reduce_sum_cols_tile(
        d2m.to_layout(tensor, input_layout), out, m_blocks, 0, grid=(2, 1)
    )
    result = out.to_host()
    for n_tile in range(1, n_tiles):
        out = d2m.empty(output_layout)
        k_reduce_sum_cols_accumulate_tile(
            d2m.to_layout(tensor, input_layout),
            d2m.to_layout(result, output_layout),
            out,
            m_blocks,
            n_tile,
            grid=(2, 1),
        )
        result = out.to_host()

    assert tuple(result.shape) == (64, 1)
    expected = tensor.sum(dim=1, keepdim=True)
    diff = (expected - result).abs().max().item()
    assert diff < 0.1, f"cross-tile reduce_sum(dim=1): max diff {diff}"


def test_reduce_max_rows_cross_tile_output_layout():
    input_layout = _make_layout(shape=(64, 64), grid_shape=(2, 2))
    output_layout = d2m.reduction_layout(input_layout, 0, allow_cross_tile=True)
    tensor = _tile_max_input((64, 64))
    m_tiles = input_layout.logical_shape[0] // 32
    n_blocks = input_layout.logical_shape[1] // 32 // 2

    out = d2m.empty(output_layout)
    k_reduce_max_rows_tile(
        d2m.to_layout(tensor, input_layout), out, 0, n_blocks, grid=(1, 2)
    )
    result = out.to_host()
    for m_tile in range(1, m_tiles):
        out = d2m.empty(output_layout)
        k_reduce_max_rows_accumulate_tile(
            d2m.to_layout(tensor, input_layout),
            d2m.to_layout(result, output_layout),
            out,
            m_tile,
            n_blocks,
            grid=(1, 2),
        )
        result = out.to_host()

    assert tuple(result.shape) == (1, 64)
    expected = tensor.max(dim=0, keepdim=True).values
    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"cross-tile reduce_max(dim=0): max diff {diff}"


@pytest.mark.parametrize("bad_dim", [2, -3, True, False])
def test_reduce_invalid_dim_rejected(bad_dim):
    @d2m.kernel
    def k_bad_dim(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        remote_store(out_t, [0, 0], reduce_sum(x, bad_dim))

    layout = _make_layout()
    tensor = torch.zeros(32, 32, dtype=torch.float32)
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k_bad_dim(d2m.to_layout(tensor, layout), d2m.empty(layout), grid=(1, 1))

    msg = str(exc_info.value)
    if isinstance(bad_dim, bool):
        assert "expected integer literal" in msg
    else:
        assert "reduce dim must be 0/1 or -2/-1" in msg
