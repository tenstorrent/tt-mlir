# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""`__matmul__` via `linalg.generic` + `d2m.tile_matmul`.

Coverage includes single-tile matmul on a multicore grid, transpose-b
variants, and an explicit multi-K loop-carried accumulator initialized with a
kernel-body zero block.
"""

import pytest
import torch
import d2m_jit as d2m

from utils import assert_pcc


@d2m.kernel
def matmul_kernel(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            c = a @ b
            remote_store(out, [m_off + m, n_off + n], c)


@d2m.kernel
def matmul_multi_k_kernel(lhs, rhs, out, k_blocks):
    c = zeros([1, 1])
    for k in range(k_blocks):
        a = remote_load(lhs, [0, k])
        b = remote_load(rhs, [k, 0])
        c += a @ b
    remote_store(out, [0, 0], c)


@d2m.kernel
def matmul_tiled_multi_k_kernel(lhs, rhs, out, m_blocks, n_blocks, k_blocks):
    for m in range(m_blocks):
        for n in range(n_blocks):
            acc = zeros([1, 1])
            for k in range(k_blocks):
                a = remote_load(lhs, [m, k])
                b = remote_load(rhs, [k, n])
                acc += a @ b
            remote_store(out, [m, n], acc)


@d2m.kernel
def matmul_transpose_b_kernel(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    c = matmul(a, b, transpose_b=True)
    remote_store(out, [0, 0], c)


@d2m.kernel
def matmul_transpose_b_method_kernel(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    c = a.matmul(b, transpose_b=True)
    remote_store(out, [0, 0], c)


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
        d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out_d, 1, 1, grid=(2, 2)
    )
    result = out_d.to_host()
    assert tuple(result.shape) == (64, 64)
    assert result.dtype == torch.float32


def test_matmul_correctness_single_tile_multicore():
    """Per-shard 32x32 matmul: each core's shard is exactly one tile, so
    the kernel emits a single `tile_matmul` per shard. Comparing against
    a per-shard torch matmul (no inter-shard K reduction)."""
    lhs = torch.randn(64, 64, dtype=torch.float32)
    rhs = torch.randn(64, 64, dtype=torch.float32)
    L = _make_layout()
    out_d = d2m.empty(L)
    matmul_kernel(
        d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out_d, 1, 1, grid=(2, 2)
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


def test_matmul_correctness_multi_k_loop_carried_accumulator():
    torch.manual_seed(0)
    lhs = torch.randn(32, 64, dtype=torch.float32) * 0.25
    rhs = torch.randn(64, 32, dtype=torch.float32) * 0.25

    lhs_layout = d2m.Layout(
        shape=(32, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    rhs_layout = d2m.Layout(
        shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    out_layout = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    out_d = d2m.empty(out_layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = True
    try:
        matmul_multi_k_kernel(
            d2m.to_layout(lhs, lhs_layout),
            d2m.to_layout(rhs, rhs_layout),
            out_d,
            2,
            grid=(1, 1),
        )
        assert_pcc(lhs @ rhs, out_d.to_host(), threshold=0.99)
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul


def test_matmul_correctness_tiled_mnk_loop_carried_accumulator():
    torch.manual_seed(0)
    lhs = torch.randn(64, 96, dtype=torch.float32) * 0.125
    rhs = torch.randn(96, 64, dtype=torch.float32) * 0.125

    lhs_layout = d2m.Layout(
        shape=(64, 96), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    rhs_layout = d2m.Layout(
        shape=(96, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    out_layout = d2m.Layout(
        shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    out_d = d2m.empty(out_layout)

    old_use_tile_matmul = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = True
    try:
        matmul_tiled_multi_k_kernel(
            d2m.to_layout(lhs, lhs_layout),
            d2m.to_layout(rhs, rhs_layout),
            out_d,
            2,
            2,
            3,
            grid=(1, 1),
        )
        assert_pcc(lhs @ rhs, out_d.to_host(), threshold=0.99)
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul


_TRANSPOSE_B_CASES = [
    pytest.param(
        (32, 32, 32),
        (1, 1),
        (1, 1),
        (1, 1),
        d2m.float32,
        torch.float32,
        0.99,
        id="f32-single-tile-32x32x32",
    ),
    pytest.param(
        (32, 32, 64),
        (1, 1),
        (2, 1),
        (1, 2),
        d2m.float32,
        torch.float32,
        0.99,
        id="f32-wide-n-32x32x64",
    ),
    pytest.param(
        (32, 64, 96),
        (1, 2),
        (3, 2),
        (1, 3),
        d2m.float32,
        torch.float32,
        0.99,
        id="f32-multi-k-wide-n-32x64x96",
    ),
    pytest.param(
        (64, 96, 64),
        (2, 3),
        (2, 3),
        (2, 2),
        d2m.float32,
        torch.float32,
        0.99,
        id="f32-tall-multi-k-64x96x64",
    ),
    pytest.param(
        (64, 64, 32),
        (2, 2),
        (1, 2),
        (2, 1),
        d2m.bfloat16,
        torch.bfloat16,
        0.96,
        id="bf16-tall-64x64x32",
    ),
]


def _constrained_rand(shape, dtype):
    if dtype == torch.float32:
        return torch.rand(shape, dtype=dtype) * 0.999 + 0.001
    return torch.rand(shape, dtype=dtype)


def _transpose_b_layouts(
    shape, lhs_block_shape, rhs_block_shape, out_block_shape, dtype
):
    m, k, n = shape
    lhs_layout = d2m.Layout(
        shape=(m, k),
        dtype=dtype,
        block_shape=list(lhs_block_shape),
        grid_shape=[1, 1],
    )
    rhs_layout = d2m.Layout(
        shape=(n, k),
        dtype=dtype,
        block_shape=list(rhs_block_shape),
        grid_shape=[1, 1],
    )
    out_layout = d2m.Layout(
        shape=(m, n),
        dtype=dtype,
        block_shape=list(out_block_shape),
        grid_shape=[1, 1],
    )
    return lhs_layout, rhs_layout, out_layout


def _run_transpose_b_case(
    kernel,
    shape,
    lhs_block_shape,
    rhs_block_shape,
    out_block_shape,
    d2m_dtype,
    torch_dtype,
    pcc,
):
    m, k, n = shape
    lhs = _constrained_rand((m, k), torch_dtype)
    rhs = _constrained_rand((n, k), torch_dtype)
    lhs_layout, rhs_layout, out_layout = _transpose_b_layouts(
        shape, lhs_block_shape, rhs_block_shape, out_block_shape, d2m_dtype
    )
    out_d = d2m.empty(out_layout)
    kernel(
        d2m.to_layout(lhs, lhs_layout),
        d2m.to_layout(rhs, rhs_layout),
        out_d,
        grid=(1, 1),
    )
    result = out_d.to_host()
    expected = lhs.to(torch.float32) @ rhs.to(torch.float32).T

    assert tuple(result.shape) == (m, n)
    assert_pcc(expected, result.to(torch.float32), threshold=pcc)


@pytest.mark.parametrize(
    "shape,lhs_block_shape,rhs_block_shape,out_block_shape,d2m_dtype,torch_dtype,pcc",
    _TRANSPOSE_B_CASES,
)
def test_matmul_transpose_b_correctness(
    shape,
    lhs_block_shape,
    rhs_block_shape,
    out_block_shape,
    d2m_dtype,
    torch_dtype,
    pcc,
):
    _run_transpose_b_case(
        matmul_transpose_b_kernel,
        shape,
        lhs_block_shape,
        rhs_block_shape,
        out_block_shape,
        d2m_dtype,
        torch_dtype,
        pcc,
    )


def test_matmul_transpose_b_method_form_correctness():
    _run_transpose_b_case(
        matmul_transpose_b_method_kernel,
        (32, 64, 32),
        (1, 2),
        (1, 2),
        (1, 1),
        d2m.float32,
        torch.float32,
        0.99,
    )


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
# `lhs[cy * M + m, k]` from the column-0 source core (cy, 0) across
# the row, and for each k, m, n it column-multicasts
# `rhs[k, cx * N + n]` from the row-0 source core (0, cx) down the
# column. The body stores `lhs_shard + rhs_shard`
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
                lhs, [cy * M + m, k], mcast_start_index=[cy, 0], mcast_shape=[1, GX]
            )
            for n in range(N):
                rhs_shard = remote_load(
                    rhs, [k, cx * N + n], mcast_start_index=[0, cx], mcast_shape=[GY, 1]
                )
                out_shard = lhs_shard + rhs_shard
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
