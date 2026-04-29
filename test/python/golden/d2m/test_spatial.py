# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple

from ttmlir.dialects import arith, d2m, tensor
from ttmlir.ir import (
    AffineConstantExpr,
    AffineDimExpr,
    AffineMap,
    AffineMapAttr,
    IndexType,
    RankedTensorType,
    Type,
)

from builder.base.builder_utils import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("d2m")

# Spatial tests use a single logical core for the inner matmul generic (grid 1x1,
# block_factors 1,1,1). Metal tensor grids match that; tile counts follow 32-tile
# geometry from the full tensor shapes.


def _build_virtual_grid_attrs(
    ctx, tensor_rank: int, core_start: Tuple[int, int]
) -> Tuple[AffineMapAttr, AffineMapAttr]:
    assert tensor_rank >= 2, f"Expected tensor rank >= 2, got {tensor_rank}"

    start_x, start_y = core_start
    d0 = AffineDimExpr.get(0, ctx)
    d1 = AffineDimExpr.get(1, ctx)
    c0 = AffineConstantExpr.get(0, ctx)
    cx = AffineConstantExpr.get(start_x, ctx)
    cy = AffineConstantExpr.get(start_y, ctx)

    # Offset only the virtual-grid axes; pass remaining axes through unchanged.
    vgm_fwd_results = [d0 + cx, d1 + cy]
    for i in range(2, tensor_rank):
        vgm_fwd_results.append(AffineDimExpr.get(i, ctx))
    vgm_fwd = AffineMap.get(tensor_rank, 0, vgm_fwd_results, ctx)

    # Physical -> virtual mapping for 2D core coordinates.
    vgm_inv = AffineMap.get(2, 0, [c0, d0 - cx, d1 - cy], ctx)
    return AffineMapAttr.get(vgm_inv), AffineMapAttr.get(vgm_fwd)


def _to_layout_with_virtual_grid_empty(
    builder: D2MBuilder,
    input_tensor: Operand,
    output_type: Type,
    virtual_grid_inverse_mapping: Optional[AffineMapAttr] = None,
    virtual_grid_forward_mapping: Optional[AffineMapAttr] = None,
    unit_attrs: Optional[List[str]] = None,
) -> Operand:
    def organize_to_layout_args(inputs, output, output_shape):
        return ([output_type], inputs[0], output)

    def output_create_fn(shape, tensor_type):
        return d2m.empty(
            tensor_type,
            virtual_grid_inverse_mapping=virtual_grid_inverse_mapping,
            virtual_grid_forward_mapping=virtual_grid_forward_mapping,
        )

    return builder._op_proxy(
        d2m.ToLayoutOp,
        [input_tensor],
        unit_attrs=unit_attrs,
        organize_d2m_args=organize_to_layout_args,
        output_type=output_type,
        output_shape=output_type.shape,
        output_create_fn=output_create_fn,
    )


def prepare_metal_input(
    builder: D2MBuilder,
    input_tensor: Operand,
    metal_type: Type,
    virtual_grid_inverse_mapping: Optional[AffineMapAttr] = None,
    virtual_grid_forward_mapping: Optional[AffineMapAttr] = None,
    unit_attrs: Optional[List[str]] = None,
) -> Operand:
    return _to_layout_with_virtual_grid_empty(
        builder,
        input_tensor,
        metal_type,
        virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping,
        unit_attrs=unit_attrs,
    )


def prepare_metal_output(
    out_metal_ty: Type,
    virtual_grid_inverse_mapping: Optional[AffineMapAttr] = None,
    virtual_grid_forward_mapping: Optional[AffineMapAttr] = None,
):
    return d2m.empty(
        out_metal_ty,
        virtual_grid_inverse_mapping=virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping=virtual_grid_forward_mapping,
    )


def matmul_region_build(
    builder: D2MBuilder,
    lhs: Operand,
    rhs: Operand,
    out: Operand,
    out_block_shape: Optional[List[int]] = None,
) -> Callable[[], None]:
    def _build():
        inner_grid = (1, 1)
        block_factors = (1, 1, 1)
        indexing_maps = (
            lambda m, n, k: (m, k),
            lambda m, n, k: (k, n),
            lambda m, n, k: (m, n),
        )
        iterator_types = ["parallel", "parallel", "reduction"]

        @builder.generic(
            grid=inner_grid,
            block_factors=block_factors,
            indexing_maps=indexing_maps,
            iterator_types=iterator_types,
        )
        def mm(lhs, rhs, out):
            mm_out_block_shape = (
                out_block_shape if out_block_shape is not None else [2, 2]
            )
            mbi = d2m.block_index(0)
            nbi = d2m.block_index(1)
            kbi = d2m.block_index(2)
            r = arith.constant(IndexType.get(lhs.context), 0)
            c = arith.constant(IndexType.get(lhs.context), 1)
            lhs_shard = builder.remote_load(lhs, [mbi, kbi], mcast_dims=[r])
            rhs_shard = builder.remote_load(rhs, [kbi, nbi], mcast_dims=[c])
            out_shard = tensor.empty(mm_out_block_shape, out.type.element_type)
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

        d2m.spatial_yield([mm(lhs, rhs, out)])

    return _build


def all_gather_region_build(
    _builder: D2MBuilder,
    _input_tensor: Operand,
    _out: Operand,
    *,
    gather_dim: int = 0,
) -> Callable[[], None]:
    """Stub: define one spatial region for all_gather (remote load/store, mesh dim)."""

    def _build():
        raise NotImplementedError(
            f"all_gather_region_build: not implemented (gather_dim={gather_dim})."
        )

    return _build


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,out_shape",
    [
        pytest.param(
            [64, 128],
            [128, 64],
            [64, 64],
            id="multi_tile",
        ),
        pytest.param(
            [32, 32],
            [32, 32],
            [32, 32],
            id="single_tile",
        ),
    ],
)
@pytest.mark.parametrize(
    "grid_ranges",
    [
        pytest.param([((0, 0), (0, 0)), ((1, 1), (1, 1))], id="basic"),
        pytest.param([((0, 0), (0, 0)), ((0, 1), (0, 1))], id="offset_x"),
        pytest.param([((1, 1), (1, 1)), ((2, 2), (2, 2))], id="none_from_origin"),
        pytest.param([((0, 0), (0, 1)), ((1, 0), (1, 1))], id="multi_cores_per_region"),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        "ttmetal",
    ],
)
def test_spatial_two_regions_two_matmuls(
    target: str,
    request,
    device,
    lhs_shape,
    rhs_shape,
    out_shape,
    grid_ranges,
):
    pipeline_opts = [
        "use-tile-matmul=false",
        "enable-l1-acc=true",
    ]
    metal_grid = [1, 1]
    out_block_tiles = [lhs_shape[0] // 32, rhs_shape[1] // 32]
    torch_dtype = torch.bfloat16

    def spatial_module(builder: D2MBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch_dtype, torch_dtype])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            ctx = lhs.context
            host_out_ty = RankedTensorType.get(out_shape, lhs.type.element_type)
            lhs_metal_ty = builder.get_metal_tensor_layout(
                lhs_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            rhs_metal_ty = builder.get_metal_tensor_layout(
                rhs_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            out_metal_ty = builder.get_metal_tensor_layout(
                out_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            lhs_metal_ty_b = builder.get_metal_tensor_layout(
                lhs_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            rhs_metal_ty_b = builder.get_metal_tensor_layout(
                rhs_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            out_metal_ty_b = builder.get_metal_tensor_layout(
                out_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            r0_start = grid_ranges[0][0]
            r1_start = grid_ranges[1][0]

            r0_vg_inv_attr, r0_vg_fwd_attr = _build_virtual_grid_attrs(
                ctx, tensor_rank=len(lhs_metal_ty.shape), core_start=r0_start
            )

            r1_vg_inv_attr, r1_vg_fwd_attr = _build_virtual_grid_attrs(
                ctx, tensor_rank=len(lhs_metal_ty_b.shape), core_start=r1_start
            )

            lhs_m = prepare_metal_input(
                builder,
                lhs,
                lhs_metal_ty,
                r0_vg_inv_attr,
                r0_vg_fwd_attr,
                unit_attrs=unit_attrs,
            )
            rhs_m = prepare_metal_input(
                builder,
                rhs,
                rhs_metal_ty,
                r0_vg_inv_attr,
                r0_vg_fwd_attr,
                unit_attrs=unit_attrs,
            )
            lhs_m_b = prepare_metal_input(
                builder,
                lhs,
                lhs_metal_ty_b,
                r1_vg_inv_attr,
                r1_vg_fwd_attr,
                unit_attrs=unit_attrs,
            )
            rhs_m_b = prepare_metal_input(
                builder,
                rhs,
                rhs_metal_ty_b,
                r1_vg_inv_attr,
                r1_vg_fwd_attr,
                unit_attrs=unit_attrs,
            )
            out0_m = prepare_metal_output(
                out_metal_ty,
                r0_vg_inv_attr,
                r0_vg_fwd_attr,
            )
            out1_m = prepare_metal_output(
                out_metal_ty_b,
                r1_vg_inv_attr,
                r1_vg_fwd_attr,
            )

            region_builders = [
                matmul_region_build(
                    builder,
                    lhs_m,
                    rhs_m,
                    out0_m,
                    out_block_shape=out_block_tiles,
                ),
                matmul_region_build(
                    builder,
                    lhs_m_b,
                    rhs_m_b,
                    out1_m,
                    out_block_shape=out_block_tiles,
                ),
            ]

            spatial_results = builder.spatial(
                [lhs_m, rhs_m, lhs_m_b, rhs_m_b],
                [out0_m, out1_m],
                grid_ranges,
                region_builders,
                result_types=[out0_m.type, out1_m.type],
                unit_attrs=unit_attrs,
            )
            r0_m, r1_m = spatial_results[0], spatial_results[1]

            res0 = builder.to_layout(
                r0_m,
                output_type=host_out_ty,
                unit_attrs=unit_attrs,
            )
            res1 = builder.to_layout(
                r1_m,
                output_type=host_out_ty,
                unit_attrs=unit_attrs,
            )

            lhs_g = torch.randn(lhs_shape, dtype=torch_dtype)
            rhs_g = torch.randn(rhs_shape, dtype=torch_dtype)
            golden = lhs_g @ rhs_g
            builder.set_goldens(
                {lhs: lhs_g, rhs: rhs_g},
                {res0: golden, res1: golden},
            )
            return (res0, res1)

    compile_and_execute_d2m(
        spatial_module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(pipeline_opts)}}}",
        print_ir=False,
        check_pcc=False,
        save_artifacts=True,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize(
    "target",
    [
        "ttmetal",
    ],
)
def test_single_matmul_offset_core(
    target: str,
    request,
    device,
):
    lhs_shape = [32, 32]
    rhs_shape = [32, 32]
    out_shape = [32, 32]
    grid_range_single_11 = [((1, 1), (1, 1))]
    pipeline_opts = [
        "use-tile-matmul=false",
        "enable-l1-acc=true",
    ]
    metal_grid = [1, 1]
    out_block_tiles = [lhs_shape[0] // 32, rhs_shape[1] // 32]

    torch_dtype = torch.bfloat16

    def spatial_module(builder: D2MBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch_dtype, torch_dtype])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            ctx = lhs.context
            host_out_ty = RankedTensorType.get(out_shape, lhs.type.element_type)
            lhs_metal_ty = builder.get_metal_tensor_layout(
                lhs_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            rhs_metal_ty = builder.get_metal_tensor_layout(
                rhs_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            out_metal_ty = builder.get_metal_tensor_layout(
                out_shape,
                tiled=True,
                element_dtype=torch_dtype,
            )
            core_start = grid_range_single_11[0][0]
            vg_inv_attr, vg_fwd_attr = _build_virtual_grid_attrs(
                ctx, tensor_rank=len(lhs_metal_ty.shape), core_start=core_start
            )

            lhs_m = prepare_metal_input(
                builder,
                lhs,
                lhs_metal_ty,
                vg_inv_attr,
                vg_fwd_attr,
                unit_attrs=unit_attrs,
            )
            rhs_m = prepare_metal_input(
                builder,
                rhs,
                rhs_metal_ty,
                vg_inv_attr,
                vg_fwd_attr,
                unit_attrs=unit_attrs,
            )
            out_m = prepare_metal_output(
                out_metal_ty,
                vg_inv_attr,
                vg_fwd_attr,
            )

            r_m = builder.spatial(
                [lhs_m, rhs_m],
                [out_m],
                grid_range_single_11,
                [
                    matmul_region_build(
                        builder,
                        lhs_m,
                        rhs_m,
                        out_m,
                        out_block_shape=out_block_tiles,
                    )
                ],
                result_types=[out_m.type],
                unit_attrs=unit_attrs,
            )

            res = builder.to_layout(
                r_m,
                output_type=host_out_ty,
                unit_attrs=unit_attrs,
            )

            lhs_g = torch.randn(lhs_shape, dtype=torch_dtype)
            rhs_g = torch.randn(rhs_shape, dtype=torch_dtype)
            golden = lhs_g @ rhs_g
            builder.set_goldens({lhs: lhs_g, rhs: rhs_g}, {res: golden})
            return res

    compile_and_execute_d2m(
        spatial_module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(pipeline_opts)}}}",
        print_ir=False,
        check_pcc=False,
        **get_request_kwargs(request),
    )
