# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttmlir.ir import Context

import d2m_jit as d2m
from d2m_jit.api import _parse_tile_bcast_type


@d2m.kernel
def k_tile_bcast_row(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], tile_bcast(x, "row"))


@d2m.kernel
def k_tile_bcast_col(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], x.tile_bcast_col())


@d2m.kernel
def k_tile_bcast_2d(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], tile_bcast_2d(x))


def make_layout():
    return d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )


def expected_tile_bcast(t, bcast_type):
    out = torch.empty_like(t)
    for tile_y in range(0, t.shape[-2], 32):
        for tile_x in range(0, t.shape[-1], 32):
            tile = t[tile_y : tile_y + 32, tile_x : tile_x + 32]
            if bcast_type == "row":
                bcast_tile = tile[:1, :].expand(32, 32)
            elif bcast_type == "col":
                bcast_tile = tile[:, :1].expand(32, 32)
            elif bcast_type == "2d":
                bcast_tile = tile[:1, :1].expand(32, 32)
            else:
                raise ValueError(f"unknown bcast type {bcast_type}")
            out[tile_y : tile_y + 32, tile_x : tile_x + 32] = bcast_tile
    return out


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("row", "row"),
        ("COL", "col"),
        ("column", "col"),
        ("2d", "scalar"),
        ("none", "none"),
        ("no_bcast", "none"),
        (d2m.TileBcastType.Row, "row"),
        (d2m.TileBcastType.Col, "col"),
        (d2m.TileBcastType.Scalar, "scalar"),
        (int(d2m.TileBcastType.Row), "row"),
        (int(d2m.TileBcastType.Col), "col"),
        (int(d2m.TileBcastType.Scalar), "scalar"),
    ],
)
def test_parse_tile_bcast_type_forms(value, expected):
    with Context():
        attr = _parse_tile_bcast_type(value)
    assert str(attr) == f"#d2m<tile_bcast_type {expected}>"


@pytest.mark.parametrize("value", ["rows", True, object()])
def test_parse_tile_bcast_type_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="tile broadcast type must be one of"):
        _parse_tile_bcast_type(value)


def test_tile_broadcasts():
    host = torch.arange(64 * 64, dtype=torch.float32).remainder(127).reshape(64, 64)
    L = make_layout()

    out_row = d2m.empty(L)
    out_col = d2m.empty(L)
    out_2d = d2m.empty(L)
    in_d = d2m.to_layout(host, L)
    k_tile_bcast_row(in_d, out_row, 1, 1, grid=(2, 2))
    k_tile_bcast_col(in_d, out_col, 1, 1, grid=(2, 2))
    k_tile_bcast_2d(in_d, out_2d, 1, 1, grid=(2, 2))

    row, col, bcast_2d = d2m.to_host(out_row, out_col, out_2d)
    assert torch.allclose(row, expected_tile_bcast(host, "row"))
    assert torch.allclose(col, expected_tile_bcast(host, "col"))
    assert torch.allclose(bcast_2d, expected_tile_bcast(host, "2d"))
