# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end round-trip + builder lifecycle:
to_layout / to_host (single + multi); spent-tensor raises; materialised
LazyTensor auto-rematerialises on re-use."""

import torch
import d2m_jit as d2m


def make_layout():
    return d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )


def test_round_trip_single():
    t = torch.randn(64, 64, dtype=torch.float32)
    out = d2m.to_layout(t, make_layout()).to_host()
    assert torch.allclose(t, out), "single-tensor round-trip mismatch"


def test_round_trip_multi():
    L = make_layout()
    t1 = torch.randn(64, 64, dtype=torch.float32)
    t2 = torch.randn(64, 64, dtype=torch.float32)
    a = d2m.to_layout(t1, L)
    b = d2m.to_layout(t2, L)
    out_a, out_b = d2m.to_host(a, b)
    assert torch.allclose(t1, out_a)
    assert torch.allclose(t2, out_b)


def test_spent_tensor_raises():
    L = make_layout()
    t1 = torch.randn(64, 64, dtype=torch.float32)
    t2 = torch.randn(64, 64, dtype=torch.float32)
    a = d2m.to_layout(t1, L)
    b = d2m.to_layout(t2, L)
    _ = a.to_host()  # materialises a, b becomes stale
    try:
        b.to_host()
    except RuntimeError as e:
        assert "Stale" in str(e), f"unexpected error msg: {e}"
        return
    raise AssertionError("stale unmaterialised LazyTensor did not raise")


def test_auto_rematerialise():
    L = make_layout()
    t = torch.randn(64, 64, dtype=torch.float32)
    a = d2m.to_layout(t, L)
    out1 = a.to_host()
    # Re-using a now auto-re-to_layouts.
    out2 = a.to_host()
    assert torch.allclose(t, out1)
    assert torch.allclose(t, out2)
