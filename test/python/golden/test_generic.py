# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pathlib import Path
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


def read_artifact(output_root: Path, test_base: str, filename: str) -> str:
    artifact = output_root / "builder-artifacts" / "D2MBuilder" / test_base / filename
    return artifact.read_text()


@pytest.mark.parametrize(
    "grid",
    [
        # (1, 1),
        (8, 8)
        | OnlyIf("n150", "n300"),
    ],
)
@pytest.mark.parametrize(
    "block_shape,block_factors",
    [
        # ((256, 256, 256), (1, 1, 1)),
        # ((64, 64, 64*8), (1, 1, 1)),
        ((64, 64, 64), (1, 1, 8)),
        # ((32, 64, 64), (2, 1, 8)),
        # ((64, 64, 32), (1, 1, 16)),
        # ((32, 64, 64), (2, 1, 8)),
    ],
)
@pytest.mark.parametrize(
    "interchange",
    [
        "mnk",
        # "kmn",
    ],
)
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("enable_l1_acc", [True])
@pytest.mark.parametrize("use_tile_matmul", [False])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_generic(
    grid,
    block_shape,
    block_factors,
    interchange,
    dtype,
    enable_l1_acc,
    use_tile_matmul,
    target: str,
    request,
    device,
    system_desc,
):
    block_m, block_n, block_k = block_shape
    m = block_m * grid[0] * block_factors[0]
    n = block_n * grid[1] * block_factors[1]
    k = block_k * block_factors[2]

    lhs_shape = [m, k]
    rhs_shape = [k, n]
    out_shape = [m, n]

    lhs_k_physical_grid = greatest_physical_grid(system_desc, 1, block_factors[2])
    rhs_k_physical_grid = greatest_physical_grid(system_desc, 0, block_factors[2])
    lhs_grid = [grid[0], lhs_k_physical_grid]
    rhs_grid = [rhs_k_physical_grid, grid[1]]
    out_grid = [grid[0], grid[1]]

    lhs_blocked_grid = [grid[0] * block_factors[0], block_factors[2]]
    rhs_blocked_grid = [block_factors[2], grid[1] * block_factors[1]]
    out_blocked_grid = [grid[0] * block_factors[0], grid[1] * block_factors[1]]

    lhs_block_shape = [block_m // 32, block_k // 32]
    rhs_block_shape = [block_k // 32, block_n // 32]
    out_block_shape = [block_m // 32, block_n // 32]

    indexing_maps, iterator_types, interchange_block_factors = {
        "mnk": (
            [
                lambda m, n, k: (m, k),
                lambda m, n, k: (k, n),
                lambda m, n, k: (m, n),
            ],
            ["parallel", "parallel", "reduction"],
            (block_factors[0], block_factors[1], block_factors[2]),
        ),
        "kmn": (
            [
                lambda k, m, n: (m, k),
                lambda k, m, n: (k, n),
                lambda k, m, n: (m, n),
            ],
            ["reduction", "parallel", "parallel"],
            (block_factors[2], block_factors[0], block_factors[1]),
        ),
    }[interchange]

    torch_dtype = {
        "f32": torch.float,
        "bf16": torch.bfloat16,
    }[dtype]

    def generic_module(builder: D2MBuilder):
        lhs_golden = torch.randn(lhs_shape, dtype=torch_dtype)
        rhs_golden = torch.randn(rhs_shape, dtype=torch_dtype)
        out_golden = lhs_golden @ rhs_golden

        @builder.func([lhs_shape, rhs_shape], [torch_dtype, torch_dtype])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            @builder.generic(
                grid=grid,
                block_factors=interchange_block_factors,
                indexing_maps=indexing_maps,
                iterator_types=iterator_types,
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
                    out.type,
                    out,
                    [mbi, nbi],
                    start_device=[],
                    device_mcast_shape=[],
                    semaphore_indices=[],
                    local_buffer=out_shard,
                )
                d2m.yield_([res])

            device_lhs = builder.to_layout(
                lhs,
                output_type=builder.get_metal_tensor_layout(
                    lhs.type.shape, grid=lhs_grid, tiled=True, element_dtype=torch_dtype
                ),
                unit_attrs=unit_attrs,
            )
            device_lhs = builder.reblock(
                device_lhs, lhs_blocked_grid, unit_attrs=unit_attrs
            )

            device_rhs = builder.to_layout(
                rhs,
                output_type=builder.get_metal_tensor_layout(
                    rhs.type.shape, grid=rhs_grid, tiled=True, element_dtype=torch_dtype
                ),
                unit_attrs=unit_attrs,
            )
            device_rhs = builder.reblock(device_rhs, rhs_blocked_grid)

            device_out = d2m.empty(
                builder.get_metal_tensor_layout(
                    out_shape, grid=out_grid, tiled=True, element_dtype=torch_dtype
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

    options = [
        f"use-tile-matmul={use_tile_matmul}",
        f"enable-l1-acc={enable_l1_acc}",
    ]
    compile_and_execute_d2m(
        generic_module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        print_ir=True,
        check_pcc=True,
        **get_request_kwargs(request),
    )
    print("Theoretical:", system_desc.calc_fpu_tops(m * n * k, units="us"))


@pytest.mark.parametrize("grid", [(1, 1)])
@pytest.mark.parametrize("block_shape", [(256, 256, 256)])
@pytest.mark.parametrize("block_factors", [(1, 1, 1)])
@pytest.mark.parametrize(
    "buffer_policy, expected_cb_shapes, unexpected_cb_shape",
    [
        (
            "max",
            ["memref<8x8x!ttcore.tile<32x32, bf16>"],
            "memref<8x1x!ttcore.tile<32x32, bf16>",
        ),
        (
            "auto",
            [
                "memref<8x1x!ttcore.tile<32x32, bf16>",
                "memref<1x8x!ttcore.tile<32x32, bf16>",
            ],
            "memref<8x8x!ttcore.tile<32x32, bf16>",
        ),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal" | OnlyIf("n150", "n300")])
def test_generic_allocator_reblock_policy(
    grid,
    block_shape,
    block_factors,
    buffer_policy,
    expected_cb_shapes,
    unexpected_cb_shape,
    target: str,
    tmp_path,
    device,
):
    torch_dtype = torch.bfloat16
    block_m, block_n, block_k = block_shape
    m = block_m * grid[0] * block_factors[0]
    n = block_n * grid[1] * block_factors[1]
    k = block_k * block_factors[2]

    lhs_shape = [m, k]
    rhs_shape = [k, n]
    out_shape = [m, n]
    out_block_shape = [block_m // 32, block_n // 32]
    test_base = f"test_generic_allocator_reblock_policy_{buffer_policy}"

    def generic_module(builder: D2MBuilder):
        lhs_golden = torch.randn(lhs_shape, dtype=torch_dtype)
        rhs_golden = torch.randn(rhs_shape, dtype=torch_dtype)
        out_golden = lhs_golden @ rhs_golden

        @builder.func([lhs_shape, rhs_shape], [torch_dtype, torch_dtype])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            @builder.generic(
                grid=grid,
                block_factors=[1, 1, 1],
                indexing_maps=[
                    lambda m, n, k: (m, k),
                    lambda m, n, k: (k, n),
                    lambda m, n, k: (m, n),
                ],
                iterator_types=["parallel", "parallel", "reduction"],
                skip_grid_selection=True,
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
                    out.type,
                    out,
                    [mbi, nbi],
                    start_device=[],
                    device_mcast_shape=[],
                    semaphore_indices=[],
                    local_buffer=out_shard,
                )
                d2m.yield_([res])

            input_layout = builder.get_metal_tensor_layout(
                lhs.type.shape, grid=grid, tiled=True, element_dtype=torch_dtype
            )
            device_lhs = builder.to_layout(
                lhs, output_type=input_layout, unit_attrs=unit_attrs
            )
            device_rhs = builder.to_layout(
                rhs, output_type=input_layout, unit_attrs=unit_attrs
            )
            device_out = d2m.empty(
                builder.get_metal_tensor_layout(
                    out_shape, grid=grid, tiled=True, element_dtype=torch_dtype
                )
            )

            mm_out = mm(device_lhs, device_rhs, device_out)
            res = builder.to_layout(
                mm_out,
                output_type=RankedTensorType.get(out_shape, lhs.type.element_type),
                unit_attrs=unit_attrs,
            )
            builder.set_goldens({lhs: lhs_golden, rhs: rhs_golden}, {res: out_golden})
            return res

    compile_and_execute_d2m(
        generic_module,
        target=target,
        device=device,
        test_base=test_base,
        output_root=str(tmp_path),
        save_artifacts=True,
        custom_pipeline=(
            "ttir-to-ttmetal-pipeline{"
            f"test-buffer-size-policy={buffer_policy} "
            "use-tile-matmul=false enable-l1-acc=false}"
        ),
        check_pcc=True,
    )

    d2m_module = read_artifact(tmp_path, test_base, "d2m_module.mlir")
    assert "block_factors = [1, 1, 1]" in d2m_module

    compiled_mlir = read_artifact(tmp_path, test_base, "ttmetal_compiled.mlir")
    for expected_cb_shape in expected_cb_shapes:
        assert expected_cb_shape in compiled_mlir
    assert unexpected_cb_shape not in compiled_mlir
