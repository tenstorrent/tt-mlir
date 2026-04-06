# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Grid size performance sweep for calibrating parallelism vs. data-motion
heuristics in D2M grid selection.

Compile and run with tracy to collect per-config perf data:

  python -m tracy -r -v --output-folder prof/<config> \
      -m ttrt run <flatbuffer>

Or compile all configs at once (generates flatbuffers under builder-artifacts/):

  pytest test/python/golden/test_grid_perf_sweep.py \
      -m perf_sweep --save-artifacts --sys-desc <path>

Not part of normal CI. Run explicitly with -m perf_sweep.
"""

import pytest
import torch
from typing import List

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = [pytest.mark.frontend("ttir"), pytest.mark.perf_sweep]

TILE = 32

# Shapes chosen so all sweep grid sizes (1..8 per dim) divide evenly.
ELTWISE_SHAPES = [
    (8 * TILE, 8 * TILE),  # 256x256   — small, DM overhead may dominate
    (16 * TILE, 16 * TILE),  # 512x512
    (32 * TILE, 32 * TILE),  # 1024x1024 — medium
    (8 * TILE, 32 * TILE),  # 256x1024  — asymmetric (wide)
    (32 * TILE, 8 * TILE),  # 1024x256  — asymmetric (tall)
    (60 * TILE, 60 * TILE),  # 1920x1920 — largest that fits 3 tensors in L1 at 4x4
]

MATMUL_SHAPES = [
    (8 * TILE, 8 * TILE, 8 * TILE),  # 256x256x256
    (16 * TILE, 16 * TILE, 16 * TILE),  # 512x512x512
    (32 * TILE, 32 * TILE, 32 * TILE),  # 1024x1024x1024
]

# Number of chained ops for overhead-amortization tests.
CHAIN_LEN = 10

# Chain shapes: allocator needs 4 live tensor-shards for chain_add (3 for chain_exp).
# Max shape at 4x4 grid: 4*(S/4)^2*2 <= 1,395,424 => S <= 1670 (chain_add, 4 tensors)
#                        3*(S/4)^2*2 <= 1,395,424 => S <= 1928 (chain_exp, 3 tensors)
CHAIN_ADD_SHAPES = ELTWISE_SHAPES + [(52 * TILE, 52 * TILE)]  # + 1664x1664
CHAIN_EXP_SHAPES = ELTWISE_SHAPES + [(60 * TILE, 60 * TILE)]  # + 1920x1920

# Grid sizes to sweep as override-device-shape (rows, cols).
GRID_SWEEP = [
    (1, 1),
    (2, 2),
    (4, 4),
    (8, 8),
    (1, 8),
    (8, 1),
    (2, 8),
    (8, 2),
]


def _grid_id(grid):
    return f"{grid[0]}x{grid[1]}"


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


def _pipeline(grid, extra_opts=""):
    r, c = grid
    opts = f"override-device-shape={r},{c}"
    if extra_opts:
        opts = f"{extra_opts} {opts}"
    return f"ttir-to-ttmetal-pipeline{{{opts}}}"


# --- Single-op: elementwise add ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize("shape", ELTWISE_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_add(shape, grid, target, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.bfloat16, torch.bfloat16])
        def eltwise_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.add(in0, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid),
        save_artifacts=True,
        **get_request_kwargs(request),
    )


# --- Single-op: exp (higher compute/element ratio than add) ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize("shape", ELTWISE_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_exp(shape, grid, target, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.bfloat16])
        def eltwise_exp(in0: Operand, builder: TTIRBuilder):
            return builder.exp(in0)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid),
        save_artifacts=True,
        **get_request_kwargs(request),
    )


# --- Single-op: reduction sum along dim 0 (grid on reduction axis) ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize("shape", ELTWISE_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_sum_dim0(shape, grid, target, request, device):
    from test_metal_reductions import create_reductions_constrained_inputs

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, "sum", [0], keep_dim=True, dtype=torch.bfloat16
        ),
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid),
        save_artifacts=True,
        **get_request_kwargs(request),
        atol=shape[0] * 0.01,
    )


# --- Single-op: reduction sum along dim 1 (grid on non-reduction axis) ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize("shape", ELTWISE_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_sum_dim1(shape, grid, target, request, device):
    from test_metal_reductions import create_reductions_constrained_inputs

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, "sum", [1], keep_dim=True, dtype=torch.bfloat16
        ),
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid),
        save_artifacts=True,
        **get_request_kwargs(request),
        atol=shape[1] * 0.01,
    )


# --- Single-op: matmul ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize("shape", MATMUL_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul(shape, grid, target, request, device):
    from test_metal_matmul import create_matmul_constrained_inputs

    m, k, n = shape
    compile_and_execute_ttir(
        create_matmul_constrained_inputs((m, k), (k, n), torch.bfloat16),
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid, "matmul-interchange=2,0,1"),
        save_artifacts=True,
        **get_request_kwargs(request),
        pcc=0.96,
    )


# --- E2E: matmul -> add (tests grid change at producer/consumer boundary) ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize(
    "shape",
    [(16 * TILE, 16 * TILE, 16 * TILE), (32 * TILE, 32 * TILE, 32 * TILE)],
    ids=_shape_id,
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_then_add(shape, grid, target, request, device):
    m, k, n = shape
    lhs, rhs, bias = (m, k), (k, n), (m, n)

    def module(builder: TTIRBuilder):
        @builder.func(
            [lhs, rhs, bias], [torch.bfloat16, torch.bfloat16, torch.bfloat16]
        )
        def matmul_then_add(
            in0: Operand, in1: Operand, in2: Operand, builder: TTIRBuilder
        ):
            in_lhs = torch.rand(lhs, dtype=torch.bfloat16)
            in_rhs = torch.rand(rhs, dtype=torch.bfloat16)
            in_bias = torch.rand(bias, dtype=torch.bfloat16)
            builder.set_goldens(inputs={in0: in_lhs, in1: in_rhs, in2: in_bias})
            mm = builder.matmul(in0, in1)
            return builder.add(mm, in2)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid, "matmul-interchange=2,0,1"),
        save_artifacts=True,
        **get_request_kwargs(request),
        pcc=0.96,
    )


# --- Chain: N x add (amortizes fixed kernel overhead) ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize("shape", CHAIN_ADD_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_chain_add(shape, grid, target, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.bfloat16, torch.bfloat16])
        def chain_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
            x = builder.add(in0, in1)
            for _ in range(CHAIN_LEN - 1):
                x = builder.add(x, in1)
            return x

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid),
        save_artifacts=True,
        **get_request_kwargs(request),
    )


# --- Chain: N x exp (higher compute/element, amortizes fixed overhead) ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize("shape", CHAIN_EXP_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_chain_exp(shape, grid, target, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.bfloat16])
        def chain_exp(in0: Operand, builder: TTIRBuilder):
            # Clamp input to [-0.1, 0.1] so exp doesn't blow up through the chain.
            in_tensor = torch.rand(shape, dtype=torch.bfloat16) * 0.2 - 0.1
            builder.set_goldens(inputs={in0: in_tensor})
            x = builder.exp(in0)
            for _ in range(CHAIN_LEN - 1):
                x = builder.exp(x)
            return x

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid),
        save_artifacts=True,
        **get_request_kwargs(request),
        pcc=0.95,
    )


# --- Chain: N x matmul (highest arithmetic intensity) ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize(
    "shape",
    [(8 * TILE, 8 * TILE, 8 * TILE), (16 * TILE, 16 * TILE, 16 * TILE)],
    ids=_shape_id,
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_chain_matmul(shape, grid, target, request, device):
    m, k, n = shape
    assert m == k == n, "chain matmul requires square matrices"

    def module(builder: TTIRBuilder):
        @builder.func([(m, k), (k, n)], [torch.bfloat16, torch.bfloat16])
        def chain_matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
            x = builder.matmul(in0, in1)
            for _ in range(CHAIN_LEN - 1):
                x = builder.matmul(x, in1)
            return x

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid, "matmul-interchange=2,0,1"),
        save_artifacts=True,
        **get_request_kwargs(request),
        pcc=0.90,
    )


# --- E2E: reduce -> add (grid shrinks across reduction boundary) ---


@pytest.mark.parametrize("grid", GRID_SWEEP, ids=_grid_id)
@pytest.mark.parametrize("shape", ELTWISE_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_then_add(shape, grid, target, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.bfloat16, torch.bfloat16])
        def reduce_then_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
            reduced = builder.sum(in0, dim_arg=[0], keep_dim=True)
            return builder.add(reduced, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(grid),
        save_artifacts=True,
        **get_request_kwargs(request),
        atol=shape[0] * 0.01,
    )
