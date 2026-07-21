# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simulator-mode acceptance tests (Phase 1).

The torch-backed simulator is selected at import time by `D2M_JIT_SIM=1`
(see SIMULATOR_SPEC.md), so this module skips unless that backend is active.
Run with:

    D2M_JIT_SIM=1 pytest test/d2m-jit/test_simulator.py

The broader numeric suite (test_eltwise / test_matmul / test_reductions /
test_broadcasts / test_zeros_full_where / test_arange_reshape /
test_tilize_untilize / test_views / test_round_trip / ...) also passes verbatim
under `D2M_JIT_SIM=1`; those exercise the sim through the exact same kernels the
device path uses. Compiler-diagnostic tests (test_errors) and the quirky
multicast-overwrite smoke test are out of scope for the simulator by design.
"""

import os

import pytest
import torch

import d2m_jit as d2m

pytestmark = pytest.mark.skipif(
    not d2m.config.simulator,
    reason="requires the torch simulator backend (set D2M_JIT_SIM=1)",
)


def _layout(shape=(64, 64), grid=(2, 2), block=(1, 1), dtype=d2m.float32):
    return d2m.Layout(
        shape=shape, dtype=dtype, block_shape=list(block), grid_shape=list(grid)
    )


def test_backend_is_simulator():
    # Sanity: the sim host API is bound, not the MLIR builder.
    assert d2m.to_layout.__module__.endswith("sim.host")


def test_builtins_cover_syntax():
    """SIM_BUILTINS must cover every free-function kernel-body op the AST
    compiler registers, so a newly-added @syntax op can't silently fall
    through to a device-only path."""
    from d2m_jit._src.ast import D2MCompiler
    from d2m_jit._src.sim import SIM_BUILTINS

    free = {
        k
        for k in D2MCompiler._syntax
        if not k.startswith("!") and not k.startswith("__")
    }
    missing = sorted(free - set(SIM_BUILTINS))
    assert not missing, f"ops missing from the simulator: {missing}"


@d2m.kernel
def _add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a + b)


def test_eltwise_add_exact():
    L = _layout()
    lhs = torch.randn(64, 64)
    rhs = torch.randn(64, 64)
    out = d2m.empty(L)
    _add(d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out, 1, 1, grid=(2, 2))
    assert torch.equal(out.to_host(), lhs + rhs)


@d2m.kernel
def _matmul(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a @ b)


def test_matmul_per_shard():
    L = _layout()
    lhs = torch.randn(64, 64)
    rhs = torch.randn(64, 64)
    out = d2m.empty(L)
    _matmul(d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out, 1, 1, grid=(2, 2))
    result = out.to_host()
    expected = torch.zeros(64, 64)
    for gy in range(2):
        for gx in range(2):
            ly, lx = gy * 32, gx * 32
            expected[ly : ly + 32, lx : lx + 32] = (
                lhs[ly : ly + 32, lx : lx + 32] @ rhs[ly : ly + 32, lx : lx + 32]
            )
    assert (result - expected).abs().max().item() < 1e-4


@d2m.kernel
def _center(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, n_off + n], x - reduce_mean(x, 1))


def test_reduce_implicit_broadcast():
    """`x - reduce_mean(x, 1)` exercises keepdim reduce + implicit broadcast."""
    L = _layout()
    t = torch.randn(64, 64)
    out = d2m.empty(L)
    _center(d2m.to_layout(t, L), out, 1, 1, grid=(2, 2))
    result = out.to_host()
    expected = torch.empty_like(t)
    for r in range(0, 64, 32):
        for c in range(0, 64, 32):
            tile = t[r : r + 32, c : c + 32]
            expected[r : r + 32, c : c + 32] = tile - tile.mean(dim=1, keepdim=True)
    assert (result - expected).abs().max().item() < 1e-4


def _nontiled_layout(shape=(4, 4), grid=(2, 2), block=(2, 2)):
    return d2m.Layout(
        shape=shape,
        dtype=d2m.float32,
        block_shape=list(block),
        grid_shape=list(grid),
        tiled=False,
    )


def test_nontiled_view_layout_reverse():
    """Arithmetic view_layout on a non-tiled layout gathers per element (no
    tile split): a reversal along the last axis."""
    L = _nontiled_layout()
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    v = d2m.view_layout(d2m.to_layout(x, L), lambda d0, d1: (d0, 3 - d1))
    assert v.is_view is True
    assert torch.equal(v._sim_logical(), torch.flip(x, dims=[1]))


def test_nontiled_view_layout_roll():
    """Non-tiled arithmetic view_layout with a modular roll along axis 0."""
    L = _nontiled_layout()
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    v = d2m.view_layout(d2m.to_layout(x, L), lambda d0, d1: ((d0 + 2) % 4, d1))
    assert torch.equal(v._sim_logical(), torch.roll(x, shifts=2, dims=0))


def test_nontiled_view_layout_out_of_bounds():
    """A remap that leaves the physical index space is rejected."""
    L = _nontiled_layout()
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    with pytest.raises(ValueError, match="out of bounds"):
        d2m.view_layout(d2m.to_layout(x, L), lambda d0, d1: (d0, d1 + 1))._sim_logical()
