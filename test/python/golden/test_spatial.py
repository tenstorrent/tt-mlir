# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List

from ttmlir.dialects import d2m, ttcore, memref, linalg, tensor, arith
from ttmlir.ir import *

from builder.base.builder_utils import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m
from test_utils import Marks, shape_str, OnlyIf, SkipIf
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("d2m")


def greatest_physical_grid(system_desc, phys_dim_index, factor):
    assert phys_dim_index < 2
    device_grid = system_desc.get_grid_shape()
    for d in range(device_grid[phys_dim_index], 0, -1):
        if factor % d == 0:
            return d
    assert False, "Failed to find factor for {factor}"


@pytest.mark.parametrize(
    "grid",
    [
        (4, 4)
        | OnlyIf("n150", "n300"),
    ],
)
@pytest.mark.parametrize(
    "block_shape,block_factors",
    [
        ((64, 64, 64), (1, 1, 8)),
    ],
)
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("enable_l1_acc", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_spatial_two_region_matmul(
    grid,
    block_shape,
    block_factors,
    dtype,
    enable_l1_acc,
    target: str,
    request,
    device,
    system_desc,
):
    """Two-region spatial: each region does an independent matmul on a disjoint
    core range (region0 on top half of rows, region1 on bottom half)."""

    block_m, block_n, block_k = block_shape
    m = block_m * grid[0] * block_factors[0]
    n = block_n * grid[1] * block_factors[1]
    k = block_k * block_factors[2]

    # Split the M dimension across two regions
    half_grid_m = grid[0] // 2
    half_m = block_m * half_grid_m * block_factors[0]
    region_grid = (half_grid_m, grid[1])

    lhs_shape = [m, k]
    rhs_shape = [k, n]
    out_shape = [m, n]

    # Per-region shapes
    region_lhs_shape = [half_m, k]
    region_out_shape = [half_m, n]

    lhs_k_physical_grid = greatest_physical_grid(system_desc, 1, block_factors[2])
    rhs_k_physical_grid = greatest_physical_grid(system_desc, 0, block_factors[2])
    lhs_grid = [grid[0], lhs_k_physical_grid]
    rhs_grid = [rhs_k_physical_grid, grid[1]]
    out_grid = [grid[0], grid[1]]

    region_lhs_grid = [half_grid_m, lhs_k_physical_grid]
    region_out_grid = [half_grid_m, grid[1]]

    lhs_blocked_grid = [grid[0] * block_factors[0], block_factors[2]]
    rhs_blocked_grid = [block_factors[2], grid[1] * block_factors[1]]
    out_blocked_grid = [grid[0] * block_factors[0], grid[1] * block_factors[1]]

    region_lhs_blocked_grid = [half_grid_m * block_factors[0], block_factors[2]]
    region_out_blocked_grid = [
        half_grid_m * block_factors[0],
        grid[1] * block_factors[1],
    ]

    out_block_shape = [block_m // 32, block_n // 32]

    indexing_maps = [
        lambda m, n, k: (m, k),
        lambda m, n, k: (k, n),
        lambda m, n, k: (m, n),
    ]
    iterator_types = ["parallel", "parallel", "reduction"]
    interchange_block_factors = (block_factors[0], block_factors[1], block_factors[2])

    torch_dtype = {
        "f32": torch.float,
        "bf16": torch.bfloat16,
    }[dtype]

    # Core ranges: region0 gets top rows, region1 gets bottom rows
    core_ranges = [
        (0, 0, half_grid_m - 1, grid[1] - 1),
        (half_grid_m, 0, grid[0] - 1, grid[1] - 1),
    ]

    def spatial_module(builder: D2MBuilder):
        lhs_golden = torch.randn(lhs_shape, dtype=torch_dtype)
        rhs_golden = torch.randn(rhs_shape, dtype=torch_dtype)
        out_golden = lhs_golden @ rhs_golden

        # Split goldens for the two regions
        lhs_golden_r0 = lhs_golden[:half_m, :]
        lhs_golden_r1 = lhs_golden[half_m:, :]
        out_golden_r0 = out_golden[:half_m, :]
        out_golden_r1 = out_golden[half_m:, :]

        @builder.func(
            [region_lhs_shape, region_lhs_shape, rhs_shape],
            [torch_dtype, torch_dtype, torch_dtype],
        )
        def main(
            lhs_r0: Operand,
            lhs_r1: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            # Layout inputs
            device_lhs_r0 = builder.to_layout(
                lhs_r0,
                output_type=builder.get_metal_tensor_layout(
                    region_lhs_shape,
                    grid=region_lhs_grid,
                    tiled=True,
                    element_dtype=torch_dtype,
                ),
                unit_attrs=unit_attrs,
            )
            device_lhs_r0 = builder.reblock(
                device_lhs_r0, region_lhs_blocked_grid, unit_attrs=unit_attrs
            )

            device_lhs_r1 = builder.to_layout(
                lhs_r1,
                output_type=builder.get_metal_tensor_layout(
                    region_lhs_shape,
                    grid=region_lhs_grid,
                    tiled=True,
                    element_dtype=torch_dtype,
                ),
                unit_attrs=unit_attrs,
            )
            device_lhs_r1 = builder.reblock(
                device_lhs_r1, region_lhs_blocked_grid, unit_attrs=unit_attrs
            )

            device_rhs = builder.to_layout(
                rhs,
                output_type=builder.get_metal_tensor_layout(
                    rhs_shape,
                    grid=rhs_grid,
                    tiled=True,
                    element_dtype=torch_dtype,
                ),
                unit_attrs=unit_attrs,
            )
            device_rhs = builder.reblock(device_rhs, rhs_blocked_grid)

            device_out_r0 = d2m.empty(
                builder.get_metal_tensor_layout(
                    region_out_shape,
                    grid=region_out_grid,
                    tiled=True,
                    element_dtype=torch_dtype,
                )
            )
            device_out_r0 = builder.reblock(device_out_r0, region_out_blocked_grid)

            device_out_r1 = d2m.empty(
                builder.get_metal_tensor_layout(
                    region_out_shape,
                    grid=region_out_grid,
                    tiled=True,
                    element_dtype=torch_dtype,
                )
            )
            device_out_r1 = builder.reblock(device_out_r1, region_out_blocked_grid)

            def make_region(builder, lhs, rhs, out):
                @builder.generic(
                    grid=region_grid,
                    block_factors=interchange_block_factors,
                    indexing_maps=indexing_maps,
                    iterator_types=iterator_types,
                    skip_grid_selection=True,
                )
                def mm(lhs, rhs, out):
                    mbi = d2m.block_index(0)
                    nbi = d2m.block_index(1)
                    kbi = d2m.block_index(2)
                    r = arith.constant(IndexType.get(lhs.context), 0)
                    c = arith.constant(IndexType.get(lhs.context), 1)
                    lhs_shard = builder.remote_load(
                        lhs, [mbi, kbi], mcast_dims=[r]
                    )
                    rhs_shard = builder.remote_load(
                        rhs, [kbi, nbi], mcast_dims=[c]
                    )
                    out_shard = tensor.empty(
                        out_block_shape, out.type.element_type
                    )
                    d2m.tile_matmul_block(lhs_shard, rhs_shard, out_shard)
                    res = d2m.remote_store(
                        out.type, out, [mbi, nbi], local_buffer=out_shard
                    )
                    d2m.yield_([res])

                return mm(lhs, rhs, out)

            res_r0, res_r1 = builder.spatial(
                core_ranges=core_ranges,
                regions=[
                    (make_region, [device_lhs_r0, device_rhs], [device_out_r0]),
                    (make_region, [device_lhs_r1, device_rhs], [device_out_r1]),
                ],
            )

            res_r0 = builder.reblock(res_r0, region_out_grid)
            res_r0 = builder.to_layout(
                res_r0,
                output_type=RankedTensorType.get(
                    region_out_shape, lhs_r0.type.element_type
                ),
                unit_attrs=unit_attrs,
            )

            res_r1 = builder.reblock(res_r1, region_out_grid)
            res_r1 = builder.to_layout(
                res_r1,
                output_type=RankedTensorType.get(
                    region_out_shape, lhs_r1.type.element_type
                ),
                unit_attrs=unit_attrs,
            )

            builder.set_goldens(
                {lhs_r0: lhs_golden_r0, lhs_r1: lhs_golden_r1, rhs: rhs_golden},
                {res_r0: out_golden_r0, res_r1: out_golden_r1},
            )
            return res_r0, res_r1

    options = [
        f"enable-l1-acc={enable_l1_acc}",
    ]
    compile_and_execute_d2m(
        spatial_module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        print_ir=True,
        check_pcc=True,
        **get_request_kwargs(request),
    )
