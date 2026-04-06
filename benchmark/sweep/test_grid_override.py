# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Grid override sweep: fixed size, vary grid via ttir.grid_override attr.
Uses standard TTIR builder — correct PCC, data in L1.
"""

import pytest
import torch
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = [pytest.mark.frontend("ttir"), pytest.mark.perf_sweep]

TILE = 32

# (shape, grid) — pick shapes that fit 3 tensors in L1 at given grid
# L1 ~1.4MB. 3 * (S/G)^2 * 2 bytes <= 1.4MB
ADD_SWEEP = [
    # 256x256: fits all grids
    ((256, 256), [1, 1]),
    ((256, 256), [2, 2]),
    ((256, 256), [4, 4]),
    ((256, 256), [8, 8]),
    # 1024x1024: fits 4x4+ (shard=256^2*2*3=384KB)
    ((1024, 1024), [4, 4]),
    ((1024, 1024), [8, 8]),
]

# Matmul: (M, K, N), grid — grid_override applied to matmul op
# L1 needs lhs shard + rhs shard + out shard
# lhs: (M/G, K) tiles, rhs: (K, N/G) tiles, out: (M/G, N/G) tiles
MATMUL_SWEEP = [
    # 256x256x256: fits all grids
    ((256, 256, 256), [1, 1]),
    ((256, 256, 256), [2, 2]),
    ((256, 256, 256), [4, 4]),
    ((256, 256, 256), [8, 8]),
    # 512x512x512: fits 4x4+
    ((512, 512, 512), [4, 4]),
    ((512, 512, 512), [8, 8]),
    # 1024x1024x256: fits 4x4+
    ((1024, 1024, 256), [4, 4]),
    ((1024, 1024, 256), [8, 8]),
]


def _add_id(cfg):
    shape, grid = cfg
    return f"{shape[0]}x{shape[1]}-g{grid[0]}x{grid[1]}"


def _mm_id(cfg):
    (M, K, N), grid = cfg
    return f"M{M}K{K}N{N}-g{grid[0]}x{grid[1]}"


@pytest.mark.parametrize("sweep_config", ADD_SWEEP, ids=_add_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_add_grid(sweep_config, target, request, device):
    shape, grid = sweep_config

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.bfloat16, torch.bfloat16])
        def add(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.add(in0, in1, grid_override=grid)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        save_artifacts=True,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("sweep_config", MATMUL_SWEEP, ids=_mm_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_grid(sweep_config, target, request, device):
    (M, K, N), grid = sweep_config

    def module(builder: TTIRBuilder):
        @builder.func([(M, K), (K, N)], [torch.bfloat16, torch.bfloat16])
        def matmul(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.matmul(in0, in1, grid_override=grid)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        save_artifacts=True,
        **get_request_kwargs(request),
        pcc=0.96,
    )
