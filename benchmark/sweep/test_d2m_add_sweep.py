# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
D2M elementwise add grid sweep using spatial block_factors.
Fixed total tensor size, vary grid (2x2 → 8x8) with block_factors compensating.
Inputs staged in DRAM, compute streams blocks through L1.
"""

import pytest
import torch
from typing import List

from ttmlir.dialects import d2m, ttcore, tensor, linalg
from ttmlir.ir import *

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m

pytestmark = [pytest.mark.frontend("d2m"), pytest.mark.perf_sweep]

TILE = 32
DRAM_GRID = [8, 8]  # Always stage host↔DRAM at max grid to keep shards small

# (shape, grid, block_factors) — grid * bf = blocked_grid (constant per shape)
# S=1024: blocked_grid=8, shard=4x4 tiles (96KB/block)
# S=2048: blocked_grid=16, shard=4x4 tiles (96KB/block)
# S=4096: blocked_grid=16, shard=8x8 tiles (384KB/block)
ADD_SWEEP = [
    ((1024, 1024), [1, 1], [8, 8]),
    ((1024, 1024), [2, 2], [4, 4]),
    ((1024, 1024), [4, 4], [2, 2]),
    ((1024, 1024), [8, 8], [1, 1]),
    ((2048, 2048), [1, 1], [16, 16]),
    ((2048, 2048), [2, 2], [8, 8]),
    ((2048, 2048), [4, 4], [4, 4]),
    ((2048, 2048), [8, 8], [2, 2]),
    ((4096, 4096), [1, 1], [16, 16]),
    ((4096, 4096), [2, 2], [8, 8]),
    ((4096, 4096), [4, 4], [4, 4]),
    ((4096, 4096), [8, 8], [2, 2]),
]


def _sweep_id(cfg):
    shape, grid, bf = cfg
    return f"{shape[0]}x{shape[1]}-g{grid[0]}x{grid[1]}"


def _build_d2m_add(shape, grid, block_factors):
    blocked_grid = [g * b for g, b in zip(grid, block_factors)]
    shard_tiles = [shape[i] // (TILE * blocked_grid[i]) for i in range(2)]

    def module(builder: D2MBuilder):
        lhs_golden = torch.randn(shape, dtype=torch.bfloat16)
        rhs_golden = torch.randn(shape, dtype=torch.bfloat16)
        out_golden = lhs_golden + rhs_golden

        @builder.func([list(shape), list(shape)], [torch.bfloat16, torch.bfloat16])
        def main(lhs: Operand, rhs: Operand, builder: D2MBuilder, unit_attrs=None):
            in_layout = builder.get_metal_tensor_layout(
                list(shape),
                grid=DRAM_GRID,
                tiled=True,
                element_dtype=torch.bfloat16,
                memorySpace=ttcore.MemorySpace.DeviceDRAM,
            )
            device_lhs = builder.to_layout(
                lhs, output_type=in_layout, unit_attrs=unit_attrs
            )
            device_rhs = builder.to_layout(
                rhs, output_type=in_layout, unit_attrs=unit_attrs
            )
            device_lhs = builder.reblock(
                device_lhs, blocked_grid, unit_attrs=unit_attrs
            )
            device_rhs = builder.reblock(
                device_rhs, blocked_grid, unit_attrs=unit_attrs
            )

            out_layout = builder.get_metal_tensor_layout(
                list(shape),
                grid=DRAM_GRID,
                tiled=True,
                element_dtype=torch.bfloat16,
                memorySpace=ttcore.MemorySpace.DeviceL1,
            )
            device_out = d2m.empty(out_layout)
            device_out_blocked = builder.reblock(device_out, blocked_grid)

            @builder.generic(
                grid=grid,
                block_factors=block_factors,
                indexing_maps=[
                    lambda m, n: (m, n),
                    lambda m, n: (m, n),
                    lambda m, n: (m, n),
                ],
                iterator_types=["parallel", "parallel"],
                skip_grid_selection=True,
            )
            def add_body(in0, in1, out):
                ctx = in0.owner.context
                mi = d2m.block_index(0)
                ni = d2m.block_index(1)
                lhs_shard = builder.remote_load(in0, [mi, ni])
                rhs_shard = builder.remote_load(in1, [mi, ni])
                out_shard = tensor.empty(shard_tiles, lhs_shard.type.element_type)
                tile_type = lhs_shard.type.element_type
                identity = AffineMap.get_identity(2, ctx)
                inner = linalg.GenericOp(
                    [out_shard.type],
                    [lhs_shard, rhs_shard],
                    [out_shard],
                    [identity] * 3,
                    [Attribute.parse('"parallel"', ctx)] * 2,
                )
                block = inner.regions[0].blocks.append(tile_type, tile_type, tile_type)
                with InsertionPoint(block):
                    result = d2m.tile_add(
                        tile_type, block.arguments[0], block.arguments[1]
                    )
                    linalg.yield_([result])
                res = d2m.remote_store(
                    out.type, out, [mi, ni], local_buffer=inner.result
                )
                d2m.yield_([res])

            mm_out = add_body(device_lhs, device_rhs, device_out_blocked)
            mm_out_phys = builder.reblock(mm_out, DRAM_GRID)
            res = builder.to_layout(
                mm_out_phys,
                output_type=RankedTensorType.get(list(shape), lhs.type.element_type),
                unit_attrs=unit_attrs,
            )
            builder.set_goldens({lhs: lhs_golden, rhs: rhs_golden}, {res: out_golden})
            return res

    return module


@pytest.mark.parametrize("sweep_config", ADD_SWEEP, ids=_sweep_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_d2m_add(sweep_config, target, request, device):
    shape, grid, block_factors = sweep_config
    r, c = grid
    compile_and_execute_d2m(
        _build_d2m_add(shape, grid, block_factors),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{enable-l1-acc=true override-device-shape={r},{c}}}",
        save_artifacts=True,
        **get_request_kwargs(request),
    )
