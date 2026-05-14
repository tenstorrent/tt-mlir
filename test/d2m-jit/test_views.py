# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""view / view_layout / permute: identity round-trips, error paths, the
is_view flag, and the to_layout(v, v.layout) materialisation escape hatch."""

import torch
import d2m_jit as d2m
import d2m_jit._src.builder as _b


def make_layout():
    return d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )


def test_view_layout_identity_round_trip():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    v = d2m.view_layout(lt, lambda d0, d1, d2, d3: (d0, d1, d2, d3))
    assert v.is_view is True, "view_layout result should be marked is_view"
    out = d2m.to_layout(v, v.layout).to_host()
    assert torch.allclose(t, out), "view_layout identity round-trip mismatch"


def test_view_identity_round_trip():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    v = d2m.view(lt, lambda d0, d1: (d0, d1))
    assert v.is_view is True
    out = d2m.to_layout(v, v.layout).to_host()
    assert torch.allclose(t, out), "view identity round-trip mismatch"


def test_view_raises_on_non_permutation():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    try:
        d2m.view(lt, lambda d0, d1: (d0, 0))
    except ValueError:
        return
    raise AssertionError("view with non-permutation lambda did not raise")


def test_view_raises_on_rank_mismatch():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    try:
        d2m.view(lt, lambda d0, d1, d2: (d0, d1, d2))
    except ValueError:
        return
    raise AssertionError("view with rank-mismatched lambda did not raise")


def test_permute_happy():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    p = d2m.permute(lt, 1, 0)
    assert p.is_view is True
    assert p.layout.logical_shape == lt.layout.logical_shape  # square
    # Identity permute should round-trip.
    _b._Builder.reset()
    lt2 = d2m.to_layout(t, make_layout())
    p_id = d2m.permute(lt2, 0, 1)
    out = d2m.to_layout(p_id, p_id.layout).to_host()
    assert torch.allclose(t, out)


def test_permute_rejects_torch_tensor():
    t = torch.zeros(64, 64)
    try:
        d2m.permute(t, 1, 0)
    except TypeError:
        return
    raise AssertionError("permute on torch.Tensor did not raise")


def test_permute_rejects_wrong_arity():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    try:
        d2m.permute(lt, 0, 1, 2)
    except ValueError:
        return
    raise AssertionError("permute(lt, 0, 1, 2) on 2D did not raise")


def test_permute_rejects_non_permutation():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    try:
        d2m.permute(lt, 0, 0)
    except ValueError:
        return
    raise AssertionError("permute(lt, 0, 0) did not raise")


def test_to_host_on_view_raises():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    v = d2m.view(lt, lambda d0, d1: (d0, d1))
    try:
        v.to_host()
    except ValueError as e:
        assert "view" in str(e).lower(), f"unexpected error msg: {e}"
        return
    raise AssertionError("to_host on view did not raise")


def test_to_layout_clears_view_flag():
    _b._Builder.reset()
    t = torch.randn(64, 64, dtype=torch.float32)
    lt = d2m.to_layout(t, make_layout())
    v = d2m.view(lt, lambda d0, d1: (d0, d1))
    mat = d2m.to_layout(v, v.layout)
    assert mat.is_view is False, "to_layout(view) should clear is_view"
