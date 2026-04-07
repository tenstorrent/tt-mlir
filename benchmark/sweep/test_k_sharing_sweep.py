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
