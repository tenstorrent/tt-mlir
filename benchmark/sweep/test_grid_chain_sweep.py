# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Grid transition tradeoff sweep: DAGs of ops where grid selection faces
competing pressures from different consumers of shared tensors.

grid_override can be:
  - None:                  let GridSelection pick automatically
  - [R, C]:                broadcast single grid to all operands of this op
  - [[R0,C0], [R1,C1], ...]: per-operand grids (inputs then output, DPS order)

Usage:
  pytest benchmark/sweep/test_grid_chain_sweep.py \
      -m perf_sweep --save-artifacts --sys-desc <path>
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
    opts = extra_opts.strip()
    if opts:
        return f"ttir-to-ttmetal-pipeline{{{opts}}}"
    return "ttir-to-ttmetal-pipeline"


def _strategy_id(strategy):
    if strategy is None:
        return "auto"
    return strategy["name"]


# ---------------------------------------------------------------------------
# Case 1: Shared tensor consumed by two matmuls with different K-dims
#
#   shared = [M, N]
#   mm1: shared[M, N] x [N, P1] -> [M, P1]    (K dim = N, streams along cols)
#   mm2: [P2, M] x shared[M, N] -> [P2, N]    (K dim = M, streams along rows)
#
# mm1 uses shared as LHS with K=N (col dim under streaming pressure).
# mm2 uses shared as RHS with K=M (row dim under streaming pressure).
# Each matmul's K-dim normalization pulls the shared tensor's grid
# along a different axis. The question: what grid for shared minimizes
# total cost (relayout + compute)?
# ---------------------------------------------------------------------------

C1_M, C1_N = 8 * TILE, 6 * TILE
C1_P1 = 7 * TILE
C1_P2 = 5 * TILE

C1_STRATEGIES = [
    None,  # auto: each matmul gets independent grid selection
    # Favor mm1's preference for shared (K=N axis)
    {
        "name": "favor_mm1",
        "mm1": [[8, 6], [6, 7], [8, 7]],
        "mm2": [[5, 8], [8, 6], [5, 6]],
    },
    # Favor mm2's preference for shared (K=M axis)
    {
        "name": "favor_mm2",
        "mm1": [[8, 6], [6, 7], [8, 7]],
        "mm2": [[5, 8], [8, 6], [5, 6]],
    },
]


@pytest.mark.parametrize("strategy", C1_STRATEGIES, ids=_strategy_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_shared_tensor_different_k_dims(strategy, target, request, device):
    """Shared tensor is LHS of mm1 (K=N) and RHS of mm2 (K=M)."""
    shared_shape = (C1_M, C1_N)
    rhs1_shape = (C1_N, C1_P1)
    lhs2_shape = (C1_P2, C1_M)

    def _grid(op_name):
        if strategy is None:
            return None
        return strategy.get(op_name)

    def module(builder: TTIRBuilder):
        @builder.func(
            [shared_shape, rhs1_shape, lhs2_shape],
            [torch.bfloat16] * 3,
        )
        def two_mm_shared(
            shared: Operand,
            rhs1: Operand,
            lhs2: Operand,
            builder: TTIRBuilder,
        ):
            t_shared = torch.rand(shared_shape, dtype=torch.bfloat16)
            t_rhs1 = torch.rand(rhs1_shape, dtype=torch.bfloat16)
            t_lhs2 = torch.rand(lhs2_shape, dtype=torch.bfloat16)
            builder.set_goldens(inputs={shared: t_shared, rhs1: t_rhs1, lhs2: t_lhs2})
            out1 = builder.matmul(shared, rhs1, grid_override=_grid("mm1"))
            out2 = builder.matmul(lhs2, shared, grid_override=_grid("mm2"))
            # Return both via add with broadcast (or just return first;
            # second may get DCE'd -- TODO: find a way to keep both live)
            return out1

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline("matmul-interchange=2,0,1"),
        save_artifacts=True,
        **get_request_kwargs(request),
        pcc=0.90,
    )


# ---------------------------------------------------------------------------
# Case 2: Shared input feeds reduction + eltwise
#
#   shared = [M, N]
#   red:  sum(shared, dim=0) -> [1, N]
#   add:  shared + bias      -> [M, N]
#
# The reduction's input grid is constrained by the reduction axis.
# The add just wants max parallelism on [M,N].
# shared can't be optimal for both simultaneously -- reduction may want
# a grid that streams well along dim 0, while add wants full block sharding.
# ---------------------------------------------------------------------------

C2_M, C2_N = 8 * TILE, 6 * TILE

C2_STRATEGIES = [
    None,
    # Force shared to use add-optimal grid for both
    {"name": "favor_add", "red": [8, 6], "add": [8, 6]},
    # Force shared to use a grid that favors reduction streaming
    {"name": "favor_red", "red": [4, 6], "add": [4, 6]},
]


@pytest.mark.parametrize("strategy", C2_STRATEGIES, ids=_strategy_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_shared_input_reduction_and_eltwise(strategy, target, request, device):
    """Shared input consumed by a reduction and an eltwise."""
    shape = (C2_M, C2_N)
    bias_shape = shape

    def _grid(op_name):
        if strategy is None:
            return None
        return strategy.get(op_name)

    def module(builder: TTIRBuilder):
        @builder.func([shape, bias_shape], [torch.bfloat16, torch.bfloat16])
        def red_and_add(shared: Operand, bias: Operand, builder: TTIRBuilder):
            t_shared = torch.rand(shape, dtype=torch.bfloat16)
            t_bias = torch.rand(bias_shape, dtype=torch.bfloat16)
            builder.set_goldens(inputs={shared: t_shared, bias: t_bias})
            red = builder.sum(
                shared, dim_arg=[0], keep_dim=True, grid_override=_grid("red")
            )
            added = builder.add(shared, bias, grid_override=_grid("add"))
            # Return the add result; reduction may get DCE'd -- TODO
            return added

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(),
        save_artifacts=True,
        **get_request_kwargs(request),
        atol=shape[0] * 0.01,
    )
