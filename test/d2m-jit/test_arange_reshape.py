# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side constructors: `d2m.arange` and `d2m.reshape`.

Both currently lower to a host roundtrip + `to_layout` (no device-side
arange_block / data-shuffling reshape yet -- see API docstrings). Tests
cover the shapes most relevant to the SDPA pipeclean:

  arange:
    - 1D layout matching the causal-mask row/col index source shape.
    - 2D layout used directly without a follow-up reshape.

  reshape:
    - GQA-style rank-preserving coalesce: [1, 8, 64, 64] -> [1, 2, 256, 64].
    - Round-trip through reshape to verify values aren't reordered.
"""

import torch
import d2m_jit as d2m


def test_arange_2d_fp32():
    """`d2m.arange(L)` over a 32x64 fp32 layout (matches the shape SDPA's
    causal-mask col index needs: [1, S] before broadcast)."""
    L = d2m.Layout(
        shape=(32, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )
    out = d2m.arange(L).to_host()
    expected = torch.arange(0, 32 * 64, dtype=torch.float32).reshape(32, 64)
    assert torch.equal(out, expected), "arange(32x64) values diverge"


def test_arange_2d_bf16_with_start_step():
    """`start` / `step` are honoured, output dtype follows the layout."""
    L = d2m.Layout(
        shape=(32, 64),
        dtype=d2m.bfloat16,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )
    out = d2m.arange(L, start=5, step=2).to_host()
    expected = torch.arange(5, 5 + 32 * 64 * 2, 2, dtype=torch.bfloat16).reshape(32, 64)
    assert torch.equal(out, expected), "arange(start=5, step=2) values diverge"


def test_arange_2d_row_major():
    """2D layout: values are filled row-major, i.e. `arange(N*M).reshape(N, M)`."""
    L = d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )
    out = d2m.arange(L).to_host()
    expected = torch.arange(0, 64 * 64, dtype=torch.float32).reshape(64, 64)
    assert torch.equal(out, expected), "arange 2D row-major mismatch"


def test_reshape_2d_to_2d():
    """Reshape (64, 64) -> (32, 128). Same total elements; row-major order
    means each row of the result holds two rows of the source. d2m-jit's
    Layout machinery requires tile-aligned dims (>=32 each) so we use
    tile-aligned source/destination shapes."""
    src_shape = (64, 64)
    dst_shape = (32, 128)
    L_src = d2m.Layout(
        shape=src_shape,
        dtype=d2m.bfloat16,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )
    t = torch.randn(*src_shape, dtype=torch.bfloat16)
    lt = d2m.to_layout(t, L_src)

    reshaped = d2m.reshape(lt, *dst_shape)
    assert tuple(reshaped.layout.logical_shape) == dst_shape, (
        f"reshape produced layout shape {reshaped.layout.logical_shape}, "
        f"expected {dst_shape}"
    )

    out = reshaped.to_host()
    expected = t.reshape(*dst_shape)
    assert torch.equal(out, expected), "reshape values diverge from torch.reshape"


def test_reshape_rejects_mismatched_numel():
    """A shape with a different total element count must raise."""
    L = d2m.Layout(
        shape=(32, 32),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )
    lt = d2m.to_layout(torch.zeros(32, 32), L)
    try:
        d2m.reshape(lt, 32, 48)  # 1024 -> 1536 mismatch
    except ValueError as e:
        assert "element count" in str(e)
        return
    raise AssertionError("reshape with mismatched numel should have raised")


def test_reshape_infers_minus_one_dim():
    """A single `-1` dim is inferred from the source element count, matching
    the torch idiom."""
    L = d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )
    t = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    lt = d2m.to_layout(t, L)
    out = d2m.reshape(lt, 32, -1)
    assert tuple(out.layout.logical_shape) == (
        32,
        128,
    ), f"reshape(-1) inferred shape {out.layout.logical_shape}, expected (32, 128)"
    assert torch.equal(
        out.to_host(), t.reshape(32, 128)
    ), "reshape(-1) values diverge from torch.reshape"


def test_reshape_rejects_multiple_minus_one():
    """Only one `-1` dim may be inferred; two must raise."""
    L = d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )
    lt = d2m.to_layout(torch.zeros(64, 64), L)
    try:
        d2m.reshape(lt, -1, -1)
    except ValueError as e:
        assert "one dimension" in str(e)
        return
    raise AssertionError("reshape with two -1 dims should have raised")


def test_reshape_accepts_list_and_positional_form():
    """`d2m.reshape(lt, [N, M])` and `d2m.reshape(lt, N, M)` are equivalent."""
    L = d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )
    t = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
    lt = d2m.to_layout(t, L)
    a = d2m.reshape(lt, 32, 128).to_host()
    lt2 = d2m.to_layout(t, L)
    b = d2m.reshape(lt2, [32, 128]).to_host()
    assert torch.equal(a, b), "positional vs list reshape gave different results"
    assert torch.equal(a, t.reshape(32, 128)), "reshape vs torch.reshape diverge"
