# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
D2M matmul grid sweep using the D2MBuilder directly.
Fixed total output size, vary grid with K-streaming block_factors.
Mirrors test_generic.py structure.
"""

import pytest
import torch
from typing import List

from ttmlir.dialects import d2m, ttcore, tensor, arith
from ttmlir.ir import *

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m

pytestmark = [pytest.mark.frontend("d2m"), pytest.mark.perf_sweep]

TILE = 32


def greatest_physical_grid(system_desc, phys_dim_index, factor):
    device_grid = system_desc.get_grid_shape()
    for d in range(device_grid[phys_dim_index], 0, -1):
        if factor % d == 0:
            return d
    assert False, f"Failed to find factor for {factor}"


# (shape_MNK, grid, block_shape, bf_k)
# M = block_m * grid[0], N = block_n * grid[1], K = block_k * bf_k
# Vary grid for same M×N×K by adjusting block_shape.
MATMUL_SWEEP = [
    # 1024x1024x1024: vary grid
    ((1024, 1024, 1024), [1, 1], (1024, 1024, 64), 16),
    ((1024, 1024, 1024), [2, 2], (512, 512, 64), 16),
    ((1024, 1024, 1024), [4, 4], (256, 256, 64), 16),
    ((1024, 1024, 1024), [8, 8], (128, 128, 64), 16),
    # 2048x2048x1024: vary grid
    ((2048, 2048, 1024), [1, 1], (2048, 2048, 64), 16),
    ((2048, 2048, 1024), [2, 2], (1024, 1024, 64), 16),
    ((2048, 2048, 1024), [4, 4], (512, 512, 64), 16),
    ((2048, 2048, 1024), [8, 8], (256, 256, 64), 16),
]


def _sweep_id(cfg):
    (M, N, K), grid, bs, bf_k = cfg
    return f"M{M}N{N}K{K}-g{grid[0]}x{grid[1]}"


@pytest.mark.parametrize("sweep_config", MATMUL_SWEEP, ids=_sweep_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_d2m_matmul(sweep_config, target, request, device, system_desc):
    (M, N, K), grid, block_shape, bf_k = sweep_config
    block_m, block_n, block_k = block_shape
    block_factors = (1, 1, bf_k)

    lhs_shape = [M, K]
    rhs_shape = [K, N]
    out_shape = [M, N]

    lhs_k_physical_grid = greatest_physical_grid(system_desc, 1, bf_k)
    rhs_k_physical_grid = greatest_physical_grid(system_desc, 0, bf_k)
    lhs_grid = [grid[0], lhs_k_physical_grid]
    rhs_grid = [rhs_k_physical_grid, grid[1]]
    out_grid = list(grid)

    lhs_blocked_grid = [grid[0] * block_factors[0], bf_k]
    rhs_blocked_grid = [bf_k, grid[1] * block_factors[1]]
    out_blocked_grid = [grid[0] * block_factors[0], grid[1] * block_factors[1]]

    out_block_shape = [block_m // TILE, block_n // TILE]

    def generic_module(builder: D2MBuilder):
        lhs_golden = torch.randn(lhs_shape, dtype=torch.bfloat16)
        rhs_golden = torch.randn(rhs_shape, dtype=torch.bfloat16)
        out_golden = lhs_golden @ rhs_golden

        @builder.func([lhs_shape, rhs_shape], [torch.bfloat16, torch.bfloat16])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            @builder.generic(
                grid=grid,
                block_factors=block_factors,
                indexing_maps=[
                    lambda m, n, k: (m, k),
                    lambda m, n, k: (k, n),
                    lambda m, n, k: (m, n),
                ],
                iterator_types=["parallel", "parallel", "reduction"],
            )
            def mm(lhs, rhs, out):
                mbi = d2m.block_index(0)
                nbi = d2m.block_index(1)
                kbi = d2m.block_index(2)
                r = arith.constant(IndexType.get(lhs.context), 0)
                c = arith.constant(IndexType.get(lhs.context), 1)
                lhs_shard = builder.remote_load(lhs, [mbi, kbi], mcast_dims=[r])
                rhs_shard = builder.remote_load(rhs, [kbi, nbi], mcast_dims=[c])
                out_shard = tensor.empty(out_block_shape, out.type.element_type)
                d2m.tile_matmul_block(lhs_shard, rhs_shard, out_shard)
                res = d2m.remote_store(
                    out.type, out, [mbi, nbi], local_buffer=out_shard
                )
                d2m.yield_([res])

            device_lhs = builder.to_layout(
                lhs,
                output_type=builder.get_metal_tensor_layout(
                    lhs.type.shape,
                    grid=lhs_grid,
                    tiled=True,
                    element_dtype=torch.bfloat16,
                ),
                unit_attrs=unit_attrs,
            )
            device_lhs = builder.reblock(
                device_lhs, lhs_blocked_grid, unit_attrs=unit_attrs
            )

            device_rhs = builder.to_layout(
                rhs,
                output_type=builder.get_metal_tensor_layout(
                    rhs.type.shape,
                    grid=rhs_grid,
                    tiled=True,
                    element_dtype=torch.bfloat16,
                ),
                unit_attrs=unit_attrs,
            )
            device_rhs = builder.reblock(device_rhs, rhs_blocked_grid)

            device_out = d2m.empty(
                builder.get_metal_tensor_layout(
                    out_shape,
                    grid=out_grid,
                    tiled=True,
                    element_dtype=torch.bfloat16,
                )
            )
            device_out = builder.reblock(device_out, out_blocked_grid)

            mm_out = mm(device_lhs, device_rhs, device_out)

            res = builder.reblock(mm_out, out_grid)
            res = builder.to_layout(
                res,
                output_type=RankedTensorType.get(out_shape, lhs.type.element_type),
                unit_attrs=unit_attrs,
            )
            builder.set_goldens({lhs: lhs_golden, rhs: rhs_golden}, {res: out_golden})
            return res

    r, c = grid
    compile_and_execute_d2m(
        generic_module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{enable-l1-acc=true override-device-shape={r},{c}}}",
        save_artifacts=True,
        **get_request_kwargs(request),
        pcc=0.96,
    )
