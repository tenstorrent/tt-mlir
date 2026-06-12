# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""`__matmul__` via `linalg.generic` + `d2m.tile_matmul`.

Two flavours covered:

1. `test_matmul_compiles_and_runs` -- baseline shape/dtype check using
   `d2m.empty` for the output. Confirms the IR lowers and runs on
   silicon. Values are not asserted because the output is uninitialised
   (the matmul body accumulates).

2. `test_matmul_correctness_via_zeros` -- pre-fill the output with
   `d2m.zeros(L)` before calling the kernel. This is the recommended
   pattern for correct values until the device-side accumulator
   zero-init lands in `_matmul_block`.
"""

import pytest
import torch
import d2m_jit as d2m
from utils import assert_pcc


# Multi-tile K-reduction expressed as a single matmul: `remote_load` fetches a
# [1,2]/[2,1] tile block (block_shape spans the K tiles) and `a @ b` reduces
# over K inside one matmul. Exercises the `@` operator on differing M x K /
# K x N operand shapes (the operator must not coerce them to a common type).
@d2m.kernel
def matmul_multi_k_kernel(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    remote_store(out, [0, 0], a @ b)


@d2m.kernel
def matmul_kernel(lhs, rhs, out, m_blocks, n_blocks, k_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            c = zeros([1, 1])
            for k in range(k_blocks):
                a = remote_load(lhs, [m_off + m, n_off + k])
                b = remote_load(rhs, [m_off + k, n_off + n])
                c += a @ b
            remote_store(out, [m_off + m, n_off + n], c)


def _make_layout():
    return d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2]
    )


def test_matmul_compiles_and_runs():
    lhs = torch.eye(64, dtype=torch.float32)
    rhs = torch.eye(64, dtype=torch.float32)
    L = _make_layout()
    out_d = d2m.empty(L)
    matmul_kernel(
        d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out_d, 1, 1, 1, grid=(2, 2)
    )
    result = out_d.to_host()
    assert tuple(result.shape) == (64, 64)
    assert result.dtype == torch.float32


def test_matmul_correctness_via_zeros():
    """Per-shard 32x32 matmul: each core's shard is exactly one tile, so
    the kernel emits a single `tile_matmul` per shard. Comparing against
    a per-shard torch matmul (no inter-shard K reduction)."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32)
    L = _make_layout()
    out_d = d2m.zeros(L)  # pre-fill accumulator
    matmul_kernel(
        d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out_d, 1, 1, 1, grid=(2, 2)
    )
    result = out_d.to_host()

    expected = torch.zeros(64, 64, dtype=torch.float32)
    for gy in range(2):
        for gx in range(2):
            ly, lx = gy * 32, gx * 32
            expected[ly : ly + 32, lx : lx + 32] = (
                lhs[ly : ly + 32, lx : lx + 32] @ rhs[ly : ly + 32, lx : lx + 32]
            )

    diff = (expected - result).abs().max().item()
    assert diff < 0.05, f"per-shard matmul: max diff {diff} too large"


def test_matmul_multi_tile_k():
    """K>1 reduction inside a single matmul: lhs/rhs `block_shape` spans the K
    tiles so one `remote_load` fetches a [1,2]/[2,1] block and `a @ b` reduces
    over K=2. This is the supported multi-tile-K form (the reduction
    accumulates in DST); it also exercises `a @ b` on differing operand
    shapes, which the `@` operator must dispatch to matmul without coercing
    the operands to a common type."""
    lhs = torch.randn(32, 64, dtype=torch.float32)
    rhs = torch.randn(64, 32, dtype=torch.float32)
    L_lhs = d2m.Layout(
        shape=(32, 64), dtype=d2m.float32, block_shape=[1, 2], grid_shape=[1, 1]
    )
    L_rhs = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[2, 1], grid_shape=[1, 1]
    )
    L_out = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    out_d = d2m.zeros(L_out)
    matmul_multi_k_kernel(
        d2m.to_layout(lhs, L_lhs), d2m.to_layout(rhs, L_rhs), out_d, grid=(1, 1)
    )
    result = out_d.to_host()
    expected = lhs @ rhs
    # PCC rather than an absolute-diff bound: f32 tile_matmul routes through the
    # SFPU's fp19, so a K-reduction accrues ~1% abs error that is correct but
    # exceeds the tight single-tile threshold used elsewhere.
    assert_pcc(expected, result)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Outer-loop matmul accumulator (`c = zeros; for k: c += a @ b`) with "
        "k_blocks > 1 computes only one K-block. The K-reduction here is an "
        "outer scf.for (not a linalg reduction), so the partials accumulate "
        "via the packer L1-acc path (llk_pack_reconfig_l1_acc). That path is "
        "emitted and reads as correctly sequenced in IR (reconfig(kk!=0) around "
        "matmul+pack), but on device the running partial is not accumulated -- "
        "an L1-acc / matmul_block sequencing issue, not a frontend or split-pass "
        "bug. The single-matmul-K form (test_matmul_multi_tile_k) works. "
        "Remove this xfail once the L1-acc outer-loop path is fixed."
    ),
)
def test_matmul_outer_loop_multi_k_xfail():
    """Repro: K=2 reduction expressed as the d2m-jit outer-loop accumulator
    (`for k: c += a @ b`). lhs/rhs grids span K so [i, k] / [k, j] address
    distinct K shards. Currently produces only one K-block's contribution."""
    lhs = torch.randn(32, 64, dtype=torch.float32)
    rhs = torch.randn(64, 32, dtype=torch.float32)
    L_lhs = d2m.Layout(
        shape=(32, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 2]
    )
    L_rhs = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1]
    )
    L_out = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    out_d = d2m.zeros(L_out)
    matmul_kernel(
        d2m.to_layout(lhs, L_lhs),
        d2m.to_layout(rhs, L_rhs),
        out_d,
        1,
        1,
        2,  # k_blocks = 2
        grid=(1, 1),
    )
    result = out_d.to_host()
    assert_pcc(lhs @ rhs, result)


# ---------------------------------------------------------------------------
# Multicast kernel
# ---------------------------------------------------------------------------
#
# Originally named `matmul` in test_simple.py but the body is `lhs + rhs`
# (not `@`) and `remote_store` overwrites `out[m, n]` on each K iteration,
# so the final value reflects only the last K. The kernel is kept as a
# smoke test for the multicast remote_load path -- both row-wise (lhs)
# and column-wise (rhs) -- on a grid larger than 1x1.


# Multicast smoke kernel.
#
# Each core (cy, cx) loops over k, m, n. For each k, m it row-multicasts
# `lhs[m, k]` from the column-0 source core (cy, 0) across the row, and
# for each k, m, n it column-multicasts `rhs[k, n]` from the row-0 source
# core (0, cx) down the column. The body stores `lhs_shard + rhs_shard`
# into `out[m, n]` -- the store is *not* accumulating, so out[m, n] ends
# up holding the last K iteration's `lhs + rhs`.
#
# (Note: a docstring inside a @d2m.kernel function body would be parsed
# as an `ast.Constant(value=str)` which the DSL's visit_Constant doesn't
# handle, so the explanation lives here as a module-level comment.)
@d2m.kernel
def mcast_overwrite_kernel(lhs, rhs, out, K, M, N, GY, GX):
    cy = core_index(0)
    cx = core_index(1)
    for k in range(K):
        for m in range(M):
            lhs_shard = remote_load(
                lhs, [m, k], mcast_start_index=[cy, 0], mcast_shape=[1, GX]
            )
            for n in range(N):
                rhs_shard = remote_load(
                    rhs, [k, n], mcast_start_index=[0, cx], mcast_shape=[GY, 1]
                )
                out_shard = lhs_shard @ rhs_shard
                remote_store(out, [m, n], out_shard)


def test_mcast_overwrite_grid_2x2():
    """Run mcast_overwrite_kernel on a 2x2 grid with K=M=N=1 -- single
    iteration per core, multicast from (cy, 0) across the row and from
    (0, cx) down the column. Verifies multicast routing dispatches
    correctly on >1x1 grid."""
    GY, GX = 2, 2
    K, M, N = 1, 1, 1

    # Each tensor is 64x64 (one 32x32 tile per core) sharded on the 2x2 grid.
    layout = d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[GY, GX],
    )

    # Distinct per-shard values so we can verify routing -- the value of
    # each output shard tells us which input shards reached it.
    lhs = torch.zeros(64, 64, dtype=torch.float32)
    rhs = torch.zeros(64, 64, dtype=torch.float32)
    for cy in range(GY):
        for cx in range(GX):
            lhs[cy * 32 : (cy + 1) * 32, cx * 32 : (cx + 1) * 32] = 10.0 * cy + cx + 1
            rhs[cy * 32 : (cy + 1) * 32, cx * 32 : (cx + 1) * 32] = 100.0 * (
                10 * cy + cx + 1
            )

    out = d2m.empty(layout)
    mcast_overwrite_kernel(
        d2m.to_layout(lhs, layout),
        d2m.to_layout(rhs, layout),
        out,
        K,
        M,
        N,
        GY,
        GX,
        grid=(GY, GX),
    )
    result = out.to_host()

    # Multicast semantics: each core (cy, cx) receives lhs's [m=0, k=0]
    # tile sourced from core (cy, 0), and rhs's [k=0, n=0] tile sourced
    # from core (0, cx). With K=M=N=1, only one iteration, so:
    #
    #   out_shard[cy, cx] = lhs_shard[(cy, 0)] + rhs_shard[(0, cx)]
    expected = torch.zeros(64, 64, dtype=torch.float32)
    for cy in range(GY):
        for cx in range(GX):
            lhs_val = 10.0 * cy + 0 + 1  # = 10*cy + 1
            rhs_val = 100.0 * (10 * 0 + cx + 1)  # = 100*(cx + 1)
            expected[cy * 32 : (cy + 1) * 32, cx * 32 : (cx + 1) * 32] = (
                lhs_val + rhs_val
            )

    assert_pcc(expected, result)
