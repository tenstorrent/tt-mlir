# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
K-dim sharing tradeoff sweep.

When two matmuls share an operand with the same K dimension, they may want
different output grids.  Grid selection normalizes K across both, but does
the shared K choice actually help or hurt?

Approach: profile isolated matmuls at different grid sizes, then profile the
combined graph.  Compare:
  (a) full grid for the wide matmul + reblock cost
  (b) reduced grid for both matmuls (no reblock)

The isolated matmul tests let us measure the raw compute cost difference
of K=10 vs K=5 parallelism.  The combined graph tests capture reblock overhead.

Usage:
  pytest benchmark/sweep/test_k_sharing_sweep.py -m perf_sweep \
      --save-artifacts --sys-desc <path>
"""

import pytest
import torch
from typing import List, Optional

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = [pytest.mark.frontend("ttir"), pytest.mark.perf_sweep]

TILE = 32


def _pipeline(extra_opts=""):
    opts = f"matmul-interchange=2,0,1"
    if extra_opts:
        opts = f"{opts} {extra_opts}"
    return f"ttir-to-ttmetal-pipeline{{{opts}}}"


def constrained_rand(shape, dtype=torch.bfloat16):
    return torch.rand(shape, dtype=dtype) * 0.999 + 0.001


# ---------------------------------------------------------------------------
# Isolated matmul at different grid sizes
#
# K = 320 (10 tiles).  At grid 10xN, block_factors K=10.
# At grid 5xN, block_factors K=5 — half the K parallelism.
# Measures the raw compute cost of reduced K.
# ---------------------------------------------------------------------------

# (lhs, rhs, description)
ISOLATED_MM = [
    # "Wide" output: benefits from large N-grid
    ((10 * TILE, 10 * TILE), (10 * TILE, 10 * TILE), "wide"),
    # "Narrow" output: less grid pressure on N
    ((10 * TILE, 10 * TILE), (10 * TILE, 2 * TILE), "narrow"),
]

GRID_SIZES = [
    (10, 10),
    (5, 10),
    (5, 5),
    (10, 2),  # narrow grid for narrow output
]


def _mm_id(cfg):
    _, _, desc = cfg
    return desc


def _grid_id(grid):
    return f"g{grid[0]}x{grid[1]}"


@pytest.mark.parametrize("mm_config", ISOLATED_MM, ids=_mm_id)
@pytest.mark.parametrize("grid", GRID_SIZES, ids=_grid_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_isolated_matmul(mm_config, grid, target, request, device):
    """Single matmul at forced grid size.  Isolates K-parallelism cost."""
    lhs_shape, rhs_shape, _ = mm_config

    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch.bfloat16, torch.bfloat16])
        def matmul(a: Operand, b: Operand, builder: TTIRBuilder):
            a_g = constrained_rand(lhs_shape)
            b_g = constrained_rand(rhs_shape)
            builder.set_goldens(inputs={a: a_g, b: b_g})
            return builder.matmul(a, b)

    r, c = grid
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(f"override-device-shape={r},{c}"),
        pcc=0.96,
        **get_request_kwargs(request),
    )


# ---------------------------------------------------------------------------
# Combined graph: W feeds wide + narrow matmuls
#
# W:[M, K] shared by:
#   mm_wide:  W @ A:[K, N_wide]  -> [M, N_wide]
#   mm_narrow: W @ B:[K, N_narrow] -> [M, N_narrow]
#
# Grid selection picks W's grid.  Wide matmul wants large grid,
# narrow matmul is fine with small grid.  One of them pays reblock.
#
# Profile at different forced device shapes to see the tradeoff:
#   dev10x10: wide gets 8x10, narrow gets 8x1 — big grid mismatch
#   dev5x5:   both get ~5x5 — no mismatch but less parallelism
# ---------------------------------------------------------------------------

# (W, A, B, description)
SHARED_CONFIGS = [
    # K=320 for both, N differs: 320 vs 64
    ((10 * TILE, 10 * TILE), (10 * TILE, 10 * TILE), (10 * TILE, 2 * TILE), "10t_wide_vs_2t_narrow"),
    # K=320, N differs more: 640 vs 32
    ((10 * TILE, 10 * TILE), (10 * TILE, 20 * TILE), (10 * TILE, 1 * TILE), "10t_wide_vs_1t_narrow"),
]


def _shared_id(cfg):
    return cfg[3]


@pytest.mark.parametrize("config", SHARED_CONFIGS, ids=_shared_id)
@pytest.mark.parametrize("grid", [(10, 10), (5, 10), (5, 5)], ids=_grid_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_shared_operand(config, grid, target, request, device):
    """Shared W feeds wide + narrow matmul.  Profile e2e including reblocks."""
    w_shape, a_shape, b_shape, _ = config
    N_wide = a_shape[1]

    def module(builder: TTIRBuilder):
        @builder.func([w_shape, a_shape, b_shape], [torch.bfloat16] * 3)
        def shared(w: Operand, a: Operand, b: Operand, builder: TTIRBuilder):
            w_g = constrained_rand(w_shape)
            a_g = constrained_rand(a_shape)
            b_g = constrained_rand(b_shape)
            builder.set_goldens(inputs={w: w_g, a: a_g, b: b_g})

            out_wide = builder.matmul(w, a)
            out_narrow = builder.matmul(w, b)

            # Keep both paths live: reduce narrow → broadcast → add to wide
            narrow_reduced = builder.sum(out_narrow, dim_arg=[-1], keep_dim=True)
            narrow_bcast = builder.broadcast(narrow_reduced, [1, N_wide])
            return builder.add(out_wide, narrow_bcast)

    r, c = grid
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(f"override-device-shape={r},{c}"),
        pcc=0.90,
        **get_request_kwargs(request),
    )


# ---------------------------------------------------------------------------
# Same shared-operand graph but with per-op grid_override
#
# This lets us directly compare "auto grid" vs "manually force compatible
# grids" on the same graph.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Size sweep — at what tensor size does grid choice start to matter?
#
# The hypothesis: for small tensors the per-op DMA overhead swamps compute,
# so different grids look similar.  For large tensors, parallelism dominates
# and a grid that halves K-parallelism noticeably hurts.
#
# Both isolated and shared variants are included.
#
# Isolated: square matmul M=K=N=S.  Sweep S across [10,20,40,60]*TILE.
#   grid (10,10) = full K-parallelism
#   grid (5, 5) = half K-parallelism
#   grid (5,10) = full N but half K
# All grid dims divide every size (all are multiples of 10).
#
# Shared: W:[S,K] → two matmuls (N_wide=S, N_narrow=2 tiles).
#   Narrow output stays small so the grid mismatch between the two consumers
#   stays structurally the same across sizes; only the scale changes.
#   Caps at 40*TILE for L1 safety with 3+ live tensors.
# ---------------------------------------------------------------------------

ISOLATED_SIZE_TILES = [10, 20, 40, 60]
SHARED_SIZE_TILES = [10, 20, 40]  # 3 live tensors; 60*TILE is ~880 KB/core at 5x5


def _size_id(s):
    return f"{s}t"


@pytest.mark.parametrize("size_tiles", ISOLATED_SIZE_TILES, ids=_size_id)
@pytest.mark.parametrize("grid", GRID_SIZES, ids=_grid_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_isolated_matmul_size_sweep(size_tiles, grid, target, request, device):
    """Square matmul M=K=N=S at varying sizes.

    Compares full-K grid (10,10) vs reduced (5,5)/(5,10) across tensor sizes
    to find where K-parallelism reduction starts costing measurable perf.
    """
    S = size_tiles * TILE
    lhs_shape = (S, S)
    rhs_shape = (S, S)

    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch.bfloat16, torch.bfloat16])
        def matmul(a: Operand, b: Operand, builder: TTIRBuilder):
            a_g = constrained_rand(lhs_shape)
            b_g = constrained_rand(rhs_shape)
            builder.set_goldens(inputs={a: a_g, b: b_g})
            return builder.matmul(a, b)

    r, c = grid
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(f"override-device-shape={r},{c}"),
        pcc=0.96,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("size_tiles", SHARED_SIZE_TILES, ids=_size_id)
@pytest.mark.parametrize("grid", [(10, 10), (5, 10), (5, 5)], ids=_grid_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_shared_operand_size_sweep(size_tiles, grid, target, request, device):
    """Shared W:[S,S] → wide (S,S) + narrow (S, 2t) matmuls at varying sizes.

    Narrow output N stays fixed at 2 tiles so the structural asymmetry between
    the two consumers is preserved across sizes.  As S grows, the reblock cost
    of forcing a compatible grid should become more visible relative to compute.
    """
    S = size_tiles * TILE
    w_shape = (S, S)
    a_shape = (S, S)         # wide consumer: N = S
    b_shape = (S, 2 * TILE)  # narrow consumer: N = 64 (fixed)
    N_wide = a_shape[1]

    def module(builder: TTIRBuilder):
        @builder.func([w_shape, a_shape, b_shape], [torch.bfloat16] * 3)
        def shared(w: Operand, a: Operand, b: Operand, builder: TTIRBuilder):
            w_g = constrained_rand(w_shape)
            a_g = constrained_rand(a_shape)
            b_g = constrained_rand(b_shape)
            builder.set_goldens(inputs={w: w_g, a: a_g, b: b_g})

            out_wide = builder.matmul(w, a)
            out_narrow = builder.matmul(w, b)

            narrow_reduced = builder.sum(out_narrow, dim_arg=[-1], keep_dim=True)
            narrow_bcast = builder.broadcast(narrow_reduced, [1, N_wide])
            return builder.add(out_wide, narrow_bcast)

    r, c = grid
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(f"override-device-shape={r},{c}"),
        pcc=0.90,
        **get_request_kwargs(request),
    )


# ---------------------------------------------------------------------------
# Hardware grid asymmetry sweep: 10 rows × 11 cols
#
# W:[K, K] is shared as RHS of mm1 and LHS of mm2:
#   mm1: X[M, K] @ W[K, K]  →  [M, K]   (W as RHS, K constrained by 10 rows)
#   mm2: W[K, K] @ Y[K, N]  →  [K, N]   (W as LHS, K constrained by 11 cols)
#
# K = n*11 tiles, so K always divides cleanly into 11 cols.  But it only
# divides cleanly into 10 rows when n is a multiple of 10 (i.e. K=110t).
#
# Grid selection choice for the shared K dimension:
#   "compat": pick the largest factor of K that fits in BOTH 10 and 11.
#             No reblock on W, but one side underutilises hardware.
#   "per_op": mm1 gets best factor ≤10, mm2 gets best factor ≤11 (always 11).
#             W must be reblocked between the two uses.
#
# | K tiles | compat | per-op     | parallelism ratio |
# |---------|--------|------------|-------------------|
# |   11    |   1    |  1 vs 11   |  11:1             |
# |   22    |   2    |  2 vs 11   |   5.5:1           |
# |   33    |   3    |  3 vs 11   |   3.7:1           |
# |   55    |   5    |  5 vs 11   |   2.2:1           |
# |  110    |  10    | 10 vs 11   |   1.1:1           |
#
# At small K: compat wastes hardware (11:1 ratio) but W is small → cheap
# reblock.  At large K: compat is nearly optimal (1.1:1) but W is huge →
# expensive reblock.  The crossover is the signal for grid selection.
#
# Sweep M and N to see how output size shifts the breakeven.
# ---------------------------------------------------------------------------

HW_ROWS, HW_COLS = 10, 11

# K values: multiples of 11 tiles.
# K=33t: compat=3, hits d2m.view_layout bug with shared compat tests.
# K=110t: W[110t,110t] exceeds L1.
# Keeping K=11t (compat=1), 22t (compat=2), 55t (compat=5).
K_HW_SWEEP_TILES = [11, 22, 55]  # 110t: W[3520,3520] exceeds L1

# M and N sweeps (tiles).
MN_SWEEP_TILES = [4, 8, 11]


def _max_factor_le(n, limit):
    """Largest factor of n that is <= limit."""
    for f in range(limit, 0, -1):
        if n % f == 0:
            return f
    return 1


def _k_hw_id(k):
    return f"K{k}t"


def _m_id(m):
    return f"M{m}t"


def _n_id(n):
    return f"N{n}t"


def _strat_hw_id(s):
    return s


# --- Isolated matmuls: measure per-op cost at compat vs optimal grid ------

@pytest.mark.parametrize("k_tiles", K_HW_SWEEP_TILES, ids=_k_hw_id)
@pytest.mark.parametrize("m_tiles", MN_SWEEP_TILES, ids=_m_id)
@pytest.mark.parametrize("strategy", ["compat", "per_op_rhs", "per_op_lhs"], ids=_strat_hw_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_hw_grid_isolated(k_tiles, m_tiles, strategy, target, request, device):
    """Single matmul with W at compat vs per-op-optimal grid.

    Tests mm1 (W as RHS) and mm2 (W as LHS) separately so we can measure
    the raw compute cost difference without reblock overhead.
    """
    K = k_tiles * TILE
    M = m_tiles * TILE
    N = m_tiles * TILE  # square output for simplicity

    compat_factor = _max_factor_le(k_tiles, min(HW_ROWS, HW_COLS))  # fits both
    rhs_factor = _max_factor_le(k_tiles, HW_ROWS)   # best for 10-row side
    lhs_factor = _max_factor_le(k_tiles, HW_COLS)    # best for 11-col side

    if strategy == "compat":
        # Both matmuls use the same grid factor for K
        grid = (compat_factor, min(m_tiles, HW_COLS))
        lhs_shape = (M, K)
        rhs_shape = (K, N)
    elif strategy == "per_op_rhs":
        # mm1: X[M,K] @ W[K,K] — K on rows, use best row factor
        grid = (rhs_factor, min(m_tiles, HW_COLS))
        lhs_shape = (M, K)
        rhs_shape = (K, N)
    else:  # per_op_lhs
        # mm2: W[K,K] @ Y[K,N] — K on cols, use best col factor
        grid = (min(m_tiles, HW_ROWS), lhs_factor)
        lhs_shape = (K, K)
        rhs_shape = (K, N)

    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch.bfloat16, torch.bfloat16])
        def matmul(a: Operand, b: Operand, builder: TTIRBuilder):
            a_g = constrained_rand(lhs_shape)
            b_g = constrained_rand(rhs_shape)
            builder.set_goldens(inputs={a: a_g, b: b_g})
            return builder.matmul(a, b)

    r, c = grid
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(f"override-device-shape={r},{c}"),
        pcc=0.80,
        **get_request_kwargs(request),
    )


# --- Shared W: compat (no reblock) vs per-op optimal (reblock W) ----------

@pytest.mark.parametrize("k_tiles", K_HW_SWEEP_TILES, ids=_k_hw_id)
@pytest.mark.parametrize("m_tiles", MN_SWEEP_TILES, ids=_m_id)
@pytest.mark.parametrize("n_tiles", MN_SWEEP_TILES, ids=_n_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_hw_grid_shared_compat(k_tiles, m_tiles, n_tiles, target, request, device):
    """Shared W at compatible grid — no reblock, potentially suboptimal.

    mm1: X[M,K] @ W[K,K] → [M,K]
    mm2: W[K,K] @ Y[K,N] → [K,N]
    Grid uses compat factor for the K-dim (fits both 10 rows and 11 cols).
    """
    K = k_tiles * TILE
    M = m_tiles * TILE
    N = n_tiles * TILE
    compat = _max_factor_le(k_tiles, min(HW_ROWS, HW_COLS))

    x_shape = (M, K)
    w_shape = (K, K)
    y_shape = (K, N)

    def module(builder: TTIRBuilder):
        @builder.func([x_shape, w_shape, y_shape], [torch.bfloat16] * 3)
        def shared(x: Operand, w: Operand, y: Operand, builder: TTIRBuilder):
            x_g = constrained_rand(x_shape)
            w_g = constrained_rand(w_shape)
            y_g = constrained_rand(y_shape)
            builder.set_goldens(inputs={x: x_g, w: w_g, y: y_g})

            out1 = builder.matmul(x, w)  # [M, K]
            out2 = builder.matmul(w, y)  # [K, N]
            return (out1, out2)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(f"override-device-shape={compat},{compat}"),
        pcc=0.85,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("k_tiles", K_HW_SWEEP_TILES, ids=_k_hw_id)
@pytest.mark.parametrize("m_tiles", MN_SWEEP_TILES, ids=_m_id)
@pytest.mark.parametrize("n_tiles", MN_SWEEP_TILES, ids=_n_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_hw_grid_shared_per_op(k_tiles, m_tiles, n_tiles, target, request, device):
    """Shared W with per-op optimal grids — reblock W, each op at best grid.

    mm1: X[M,K] @ W[K,K] → grid uses best factor of K for 10 rows
    mm2: W[K,K] @ Y[K,N] → grid uses best factor of K for 11 cols
    W is reblocked between the two matmuls.
    """
    K = k_tiles * TILE
    M = m_tiles * TILE
    N = n_tiles * TILE

    x_shape = (M, K)
    w_shape = (K, K)
    y_shape = (K, N)

    # Let the compiler pick per-op grids on full hardware
    def module(builder: TTIRBuilder):
        @builder.func([x_shape, w_shape, y_shape], [torch.bfloat16] * 3)
        def shared(x: Operand, w: Operand, y: Operand, builder: TTIRBuilder):
            x_g = constrained_rand(x_shape)
            w_g = constrained_rand(w_shape)
            y_g = constrained_rand(y_shape)
            builder.set_goldens(inputs={x: x_g, w: w_g, y: y_g})

            out1 = builder.matmul(x, w)  # [M, K]
            out2 = builder.matmul(w, y)  # [K, N]
            return (out1, out2)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(f"override-device-shape={HW_ROWS},{HW_COLS}"),
        pcc=0.85,
        **get_request_kwargs(request),
    )


OVERRIDE_STRATEGIES = [
    None,  # auto
    {
        "name": "force_wide_grid",
        "mm_wide": [8, 10],
        "mm_narrow": [8, 1],
    },
    {
        "name": "force_compat_5x5",
        "mm_wide": [5, 5],
        "mm_narrow": [5, 1],
    },
]


def _strat_id(s):
    return "auto" if s is None else s["name"]


@pytest.mark.parametrize("strategy", OVERRIDE_STRATEGIES, ids=_strat_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_shared_operand_grid_override(strategy, target, request, device):
    """Shared W with explicit per-op grid overrides vs auto."""
    w_shape = (10 * TILE, 10 * TILE)
    a_shape = (10 * TILE, 10 * TILE)
    b_shape = (10 * TILE, 2 * TILE)
    N_wide = a_shape[1]

    def _grid(op_name):
        if strategy is None:
            return None
        return strategy.get(op_name)

    def module(builder: TTIRBuilder):
        @builder.func([w_shape, a_shape, b_shape], [torch.bfloat16] * 3)
        def shared(w: Operand, a: Operand, b: Operand, builder: TTIRBuilder):
            w_g = constrained_rand(w_shape)
            a_g = constrained_rand(a_shape)
            b_g = constrained_rand(b_shape)
            builder.set_goldens(inputs={w: w_g, a: a_g, b: b_g})

            out_wide = builder.matmul(w, a, grid_override=_grid("mm_wide"))
            out_narrow = builder.matmul(w, b, grid_override=_grid("mm_narrow"))

            narrow_reduced = builder.sum(out_narrow, dim_arg=[-1], keep_dim=True)
            narrow_bcast = builder.broadcast(narrow_reduced, [1, N_wide])
            return builder.add(out_wide, narrow_bcast)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(),
        pcc=0.90,
        **get_request_kwargs(request),
    )
