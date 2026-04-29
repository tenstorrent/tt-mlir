# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import _ttmlir_runtime as tt_runtime

from builder.base.builder_apis import compile_and_execute_d2m
from builder.base.builder_utils import Operand, get_metal_tensor_layout
from builder.d2m.d2m_builder import D2MBuilder
from conftest import get_request_kwargs
from test_utils import make_shard_shape
from ttmlir.dialects import affine, arith, d2m, tensor, ttcore
from ttmlir.ir import (
    AffineConstantExpr,
    AffineDimExpr,
    AffineMap,
    AffineMapAttr,
    ArrayAttr,
    Attribute,
    Context,
    DenseElementsAttr,
    DenseI64ArrayAttr,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    RankedTensorType,
    Type,
)

pytestmark = pytest.mark.frontend("d2m")


def _greatest_physical_grid(
    physical_grid_shape: Tuple[int, int], phys_dim_index: int, factor: int
):
    assert phys_dim_index < 2
    for d in range(physical_grid_shape[phys_dim_index], 0, -1):
        if factor % d == 0:
            return d
    assert False, "Failed to find factor for {factor}"


def _compute_matmul_layout_config(
    lhs_shape: List[int],
    rhs_shape: List[int],
    grid: Tuple[int, int],
    block_factors: Tuple[int, int, int],
    core_range: Tuple[Tuple[int, int], Tuple[int, int]],
):
    block_m = lhs_shape[0] // (grid[0] * block_factors[0])
    block_n = rhs_shape[1] // (grid[1] * block_factors[1])
    out_block_tiles = [block_m // 32, block_n // 32]
    core_range_shape = (
        core_range[1][0] - core_range[0][0] + 1,
        core_range[1][1] - core_range[0][1] + 1,
    )
    lhs_k_physical_grid = _greatest_physical_grid(core_range_shape, 1, block_factors[2])
    rhs_k_physical_grid = _greatest_physical_grid(core_range_shape, 0, block_factors[2])

    return {
        "lhs_grid": [grid[0], lhs_k_physical_grid],
        "rhs_grid": [rhs_k_physical_grid, grid[1]],
        "out_grid": [grid[0], grid[1]],
        "out_block_tiles": out_block_tiles,
    }


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


def d2m_mesh_shard_devices(
    host_tensor: Operand,
    shard_result_type: RankedTensorType,
    shard_shape: Sequence[int],
    shard_dims: Sequence[int],
) -> Operand:
    """
    d2m.mesh_shard with MeshShardType.Devices and FullToShard (same contract as
    TTIRBuilder.mesh_shard in test/python/golden/d2m/test_allgather.py).

    The first operand to d2m.mesh_shard is the result tensor type; the second is
    the full host tensor to shard.
    """
    ctx = host_tensor.context
    st = ttcore.ir.MeshShardTypeAttr.get(ctx, ttcore.ir.MeshShardType.Devices)
    sd = ttcore.ir.MeshShardDirectionAttr.get(
        ctx, ttcore.ir.MeshShardDirection.FullToShard
    )
    ss = DenseI64ArrayAttr.get(list(shard_shape), ctx)
    sdims = DenseI64ArrayAttr.get(list(shard_dims), ctx)
    return d2m.mesh_shard(shard_result_type, host_tensor, st, sd, ss, sdims)


def _tensor_mesh_encoding(ctx: Context, mesh_name: str = "mesh") -> Attribute:
    """TensorMesh encoding (Python lacks ttcore.ir.TensorMeshAttr.get)."""
    probe = Type.parse(f'tensor<1x1xf32, #ttcore.tensor_mesh<"{mesh_name}">>', ctx)
    return RankedTensorType(probe).encoding


def _mesh_ranked_f32_ty(
    ctx: Context, shape: Sequence[int], mesh_name: str = "mesh"
) -> RankedTensorType:
    return RankedTensorType.get(
        list(shape), F32Type.get(ctx), _tensor_mesh_encoding(ctx, mesh_name)
    )


def d2m_mesh_shard_devices_shard_to_full(
    mesh_tensor: Operand,
    full_host_result_ty: RankedTensorType,
    shard_shape: Sequence[int],
    shard_dims: Sequence[int],
) -> Operand:
    ctx = mesh_tensor.context
    st = ttcore.ir.MeshShardTypeAttr.get(ctx, ttcore.ir.MeshShardType.Devices)
    sd = ttcore.ir.MeshShardDirectionAttr.get(
        ctx, ttcore.ir.MeshShardDirection.ShardToFull
    )
    ss = DenseI64ArrayAttr.get(list(shard_shape), ctx)
    sdims = DenseI64ArrayAttr.get(list(shard_dims), ctx)
    return d2m.mesh_shard(full_host_result_ty, mesh_tensor, st, sd, ss, sdims)


def _two_dim_collapse_intervals(ctx: Context) -> DenseElementsAttr:
    i64 = IntegerType.get_signless(64)
    collapse_ty = RankedTensorType.get([2, 2], i64)
    return DenseElementsAttr.get(
        np.array([[0, 1], [1, 2]], dtype=np.int64), type=collapse_ty
    )


def _global_semaphore_backing_tensor_type(ctx: Context) -> RankedTensorType:
    """8x8x1x1 ui32 L1 sharded backing tensor for create_global_semaphore."""
    collapse = _two_dim_collapse_intervals(ctx)
    layout = ttcore.ir.MetalLayoutAttr.get(
        ctx,
        [8, 8],
        ttcore.OOBVal.Undef,
        ttcore.MemorySpace.DeviceL1,
        ttcore.TensorMemoryLayout.Sharded,
        collapse,
        [1, 1],
    )
    u32 = IntegerType.get_unsigned(32)
    return RankedTensorType.get([8, 8, 1, 1], u32, layout)


def fabric_ring_all_gather_view_remaps(ctx: Context) -> Tuple[Attribute, Attribute]:
    """view_layout remappings for lowered fabric all_gather stream tensors.

    Must match TTIRToD2M D2MAllGatherRewriter (see working/spatial_builder/all_gather/sample.mlir).
    """
    return (
        Attribute.parse(
            "affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2 + d3 floordiv 8, d3 mod 8, 0, 0)>",
            ctx,
        ),
        Attribute.parse(
            "affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2 + (d1 * 8 + d3) floordiv 64, "
            "(d1 + d3 floordiv 8) mod 8, 0, d3 mod 8)>",
            ctx,
        ),
    )


def fabric_ring_all_gather_metal_grid(mesh_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Physical grid for stream metal tensors produced by current 1xN ring lowering."""
    n = mesh_shape[0] * mesh_shape[1]
    return (n, n)


@dataclass(frozen=True)
class FabricRingAllGatherSpatialStreams:
    """Operands wired into d2m.spatial for the fabric all_gather region (like lhs_m/rhs_m/out for matmul)."""

    view_in: Operand
    view_out: Operand
    start_sem: Operand
    end_sem: Operand


def prepare_fabric_ring_all_gather_spatial_streams(
    builder: D2MBuilder,
    ag_host_full: Operand,
    *,
    per_device_shard_logical: Tuple[int, int],
    gathered_mesh_logical: Tuple[int, int],
    mesh_shape: Tuple[int, int],
    shard_shape: Sequence[int],
    shard_dims: Sequence[int],
    view_in_shape: Tuple[int, int, int, int],
    view_out_shape: Tuple[int, int, int, int],
    mesh_name: str = "mesh",
    unit_attrs: Optional[List[str]] = None,
) -> FabricRingAllGatherSpatialStreams:
    """Host mesh_shard, semaphores, to_layout, and view_layout (mirrors matmul prepare_metal_* layering).

    Metal tensor types use get_metal_tensor_layout with ``fabric_ring_all_gather_metal_grid`` so
    shapes match TTIRToD2M without hand-written #ttcore.metal_layout strings.
    """
    ctx = ag_host_full.context
    metal_grid = fabric_ring_all_gather_metal_grid(mesh_shape)
    torch_f32 = torch.float32

    shard_mesh_ty = _mesh_ranked_f32_ty(ctx, per_device_shard_logical, mesh_name)
    ag_sharded = d2m_mesh_shard_devices(
        ag_host_full, shard_mesh_ty, shard_shape, shard_dims
    )

    sem_ty = _global_semaphore_backing_tensor_type(ctx)
    sem_gs_ty = Type.parse("!d2m.global_semaphore", ctx)
    start_sem = d2m.create_global_semaphore(
        d2m.empty(sem_ty), value=0, results=[sem_gs_ty]
    )
    end_sem = d2m.create_global_semaphore(
        d2m.empty(sem_ty), value=0, results=[sem_gs_ty]
    )

    mesh_init_ty = _mesh_ranked_f32_ty(ctx, gathered_mesh_logical, mesh_name)
    ag_mesh_init = d2m.empty(mesh_init_ty)

    tile_in_ty = get_metal_tensor_layout(
        ctx,
        list(per_device_shard_logical),
        tiled=True,
        element_dtype=torch_f32,
        grid=metal_grid,
    )
    ag_tile_in = d2m.to_layout([tile_in_ty], ag_sharded, d2m.empty(tile_in_ty))

    tile_out_ty = get_metal_tensor_layout(
        ctx,
        list(gathered_mesh_logical),
        tiled=True,
        element_dtype=torch_f32,
        grid=metal_grid,
    )
    ag_tile_out = d2m.to_layout([tile_out_ty], ag_mesh_init, d2m.empty(tile_out_ty))

    map_in, map_out = fabric_ring_all_gather_view_remaps(ctx)
    view_in_ty = RankedTensorType.get(
        list(view_in_shape), tile_in_ty.element_type, tile_in_ty.encoding
    )
    view_out_ty = RankedTensorType.get(
        list(view_out_shape), tile_out_ty.element_type, tile_out_ty.encoding
    )
    ag_view_in = d2m.view_layout(view_in_ty, ag_tile_in, map_in)
    ag_view_out = d2m.view_layout(view_out_ty, ag_tile_out, map_out)

    return FabricRingAllGatherSpatialStreams(
        view_in=ag_view_in,
        view_out=ag_view_out,
        start_sem=start_sem,
        end_sem=end_sem,
    )


def finalize_fabric_ring_all_gather_to_host(
    ag_spatial_result: Operand,
    *,
    gathered_mesh_logical: Tuple[int, int],
    host_full_ty: RankedTensorType,
    shard_shape: Sequence[int],
    shard_dims: Sequence[int],
    mesh_name: str = "mesh",
) -> Operand:
    """to_layout back to mesh-ranked tensor then mesh_shard ShardToFull (post-spatial)."""
    ctx = ag_spatial_result.context
    mesh_ty = _mesh_ranked_f32_ty(ctx, gathered_mesh_logical, mesh_name)
    ag_mesh = d2m.to_layout([mesh_ty], ag_spatial_result, d2m.empty(mesh_ty))
    return d2m_mesh_shard_devices_shard_to_full(
        ag_mesh, host_full_ty, shard_shape, shard_dims
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
    builder: D2MBuilder,
    view_in: Operand,
    view_out: Operand,
    start_sem: Operand,
    end_sem: Operand,
    *,
    num_mesh_devices: int = 8,
) -> Callable[[], None]:
    """One spatial region: fabric ring all_gather d2m.generic (sample.mlir lines 26-50).

    Callers must create global semaphores and view_layout stream tensors outside
    the region (same as TTIRToD2M D2MAllGatherRewriter). Mesh is expected 1xN with
    cluster_axis = 1 in the fabric config (see sample.mlir).
    """

    def _build():
        ctx = view_in.context
        idx_ty = IndexType.get(ctx)
        i32_ty = IntegerType.get_signless(32)
        c0 = arith.constant(idx_ty, 0)
        c1 = arith.constant(idx_ty, 1)
        wait_val = num_mesh_devices - 1
        c_wait = arith.constant(idx_ty, wait_val)
        c8 = arith.constant(idx_ty, num_mesh_devices)
        fabric = Attribute.parse(
            "#ttcore.fabric_connection_config<"
            "noc_index = noc0, topology = ring, cluster_axis = 1, "
            "routing_mode = unidir_ring_torus, num_links = 1>",
            ctx,
        )
        grid_attr = Attribute.parse("#ttcore.grid<2x1>", ctx)
        block_factors = DenseI64ArrayAttr.get([], ctx)
        empty_iter = ArrayAttr.get([], context=ctx)
        threads = ArrayAttr.get(
            [Attribute.parse("#d2m.thread<unified>", ctx)], context=ctx
        )
        d0 = AffineDimExpr.get(0, ctx)
        d1 = AffineDimExpr.get(1, ctx)
        d2 = AffineDimExpr.get(2, ctx)
        map2 = AffineMap.get(3, 0, [d1], ctx)
        map3 = AffineMap.get(3, 0, [d0 + d2], ctx)
        num_recv = num_mesh_devices - 1

        ag_generic = d2m.GenericOp(
            [view_out.type],
            [view_in],
            [view_out],
            [start_sem, end_sem],
            grid_attr,
            block_factors,
            [],
            empty_iter,
            threads,
            1,
            fabricConnectionConfig=fabric,
        )
        ag_generic.regions[0].blocks.append()
        gblock = ag_generic.regions[0].blocks[0]
        with InsertionPoint(gblock):
            mesh_row = d2m.mesh_position(0)
            core0 = d2m.core_index(0)
            core1 = d2m.core_index(1)
            d2m.device_synchronize(
                start_sem,
                [mesh_row, c0],
                [c1, c8],
                IntegerAttr.get(i32_ty, num_recv),
                [core0, core1],
            )
            core0_1 = d2m.core_index(0)
            c0_2 = arith.constant(idx_ty, 0)
            loaded = builder.remote_load(view_in, [core0_1, c0_2])
            mesh_col = d2m.mesh_position(1)
            idx_row = affine.apply(map2, [mesh_col, core0_1, c0_2])
            idx_col = affine.apply(map3, [mesh_col, core0_1, c0_2])
            stored = d2m.remote_store(
                view_out.type,
                view_out,
                [idx_row, idx_col],
                [mesh_row, c0],
                [c1, c8],
                [core0_1, c0_2],
                local_buffer=loaded,
                semaphore=end_sem,
            )
            d2m.semaphore_wait(end_sem, c_wait)
            d2m.yield_([stored])
        d2m.spatial_yield([ag_generic.result])

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
    grid = (1, 1)
    block_factors = (1, 1, 1)
    pipeline_opts = [
        "use-tile-matmul=false",
        "enable-l1-acc=true",
    ]
    layout_cfg_r0 = _compute_matmul_layout_config(
        lhs_shape,
        rhs_shape,
        grid,
        block_factors,
        core_range=grid_ranges[0],
    )
    layout_cfg_r1 = _compute_matmul_layout_config(
        lhs_shape,
        rhs_shape,
        grid,
        block_factors,
        core_range=grid_ranges[1],
    )
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
                grid=layout_cfg_r0["lhs_grid"],
                tiled=True,
                element_dtype=torch_dtype,
            )
            rhs_metal_ty = builder.get_metal_tensor_layout(
                rhs_shape,
                grid=layout_cfg_r0["rhs_grid"],
                tiled=True,
                element_dtype=torch_dtype,
            )
            out_metal_ty = builder.get_metal_tensor_layout(
                out_shape,
                grid=layout_cfg_r0["out_grid"],
                tiled=True,
                element_dtype=torch_dtype,
            )
            lhs_metal_ty_b = builder.get_metal_tensor_layout(
                lhs_shape,
                grid=layout_cfg_r1["lhs_grid"],
                tiled=True,
                element_dtype=torch_dtype,
            )
            rhs_metal_ty_b = builder.get_metal_tensor_layout(
                rhs_shape,
                grid=layout_cfg_r1["rhs_grid"],
                tiled=True,
                element_dtype=torch_dtype,
            )
            out_metal_ty_b = builder.get_metal_tensor_layout(
                out_shape,
                grid=layout_cfg_r1["out_grid"],
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
                    out_block_shape=layout_cfg_r0["out_block_tiles"],
                ),
                matmul_region_build(
                    builder,
                    lhs_m_b,
                    rhs_m_b,
                    out1_m,
                    out_block_shape=layout_cfg_r1["out_block_tiles"],
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
    "grid,block_factors,block_shape",
    [((1, 1), (1, 1, 1), (32, 32, 32))],
    ids=["1_1"],
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
    grid,
    block_factors,
    block_shape,
):
    # Match test_generic grid0 / block_shape0 / block_factors0.
    block_m, block_n, block_k = block_shape
    m = block_m * grid[0] * block_factors[0]
    n = block_n * grid[1] * block_factors[1]
    k = block_k * block_factors[2]
    lhs_shape = [m, k]
    rhs_shape = [k, n]
    out_shape = [m, n]
    grid_range_single_11 = [((1, 1), (1, 1))]
    pipeline_opts = [
        "use-tile-matmul=false",
        "enable-l1-acc=true",
    ]
    layout_cfg = _compute_matmul_layout_config(
        lhs_shape,
        rhs_shape,
        grid,
        block_factors,
        core_range=grid_range_single_11[0],
    )

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
                grid=layout_cfg["lhs_grid"],
                tiled=True,
                element_dtype=torch_dtype,
            )
            rhs_metal_ty = builder.get_metal_tensor_layout(
                rhs_shape,
                grid=layout_cfg["rhs_grid"],
                tiled=True,
                element_dtype=torch_dtype,
            )
            out_metal_ty = builder.get_metal_tensor_layout(
                out_shape,
                grid=layout_cfg["out_grid"],
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
                        out_block_shape=layout_cfg["out_block_tiles"],
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


@pytest.mark.parametrize("mesh_shape", [(1, 8)], ids=["1x8"])
@pytest.mark.parametrize(
    "fabric_config",
    [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_spatial_matmul_and_all_gather_single_tile(
    target: str,
    request,
    device,
    mesh_shape: Tuple[int, int],
    fabric_config,
):
    """Matmul on one core range plus fabric ring all_gather aligned with sample.mlir.

    The all_gather path matches post-ttir-to-d2m IR for func @all_gather on a
    1x8 mesh (256x2048 f32 per-device shard -> 256x16384 full). See
    working/spatial_builder/all_gather/sample.mlir. Matmul stays a small bf16
    single-tile case on a separate spatial region.

    Requires eight devices in system_desc (deselected otherwise). Uses ring
    fabric like test_allgather.py (mesh-topology=linear,ring).
    """
    mesh_dict = OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])])
    num_mesh_devices = mesh_shape[0] * mesh_shape[1]

    mm_shape = (32, 32)
    # Logical shapes for lowered fabric ring all_gather (sample.mlir @all_gather).
    ag_per_device_shard = (256, 256)
    ag_gathered_on_mesh = (256, 2048)
    ag_full_host = (256, 16384)
    ag_view_in_shape = (2, 1, 4, 8)
    ag_view_out_shape = (2, 8, 4, 8)
    rank_in = 2
    rank_mesh = len(mesh_shape)
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    torch_dtype_mm = torch.bfloat16
    torch_dtype_ag = torch.float32
    grid = (1, 1)
    block_factors = (1, 1, 1)
    lhs_shape = list(mm_shape)
    rhs_shape = list(mm_shape)
    out_shape = list(mm_shape)
    grid_ranges = [
        ((0, 0), (0, 0)),
        ((1, 1), (1, 1)),
    ]
    layout_cfg_r0 = _compute_matmul_layout_config(
        lhs_shape, rhs_shape, grid, block_factors, core_range=grid_ranges[0]
    )

    def spatial_module(builder: D2MBuilder):
        @builder.func(
            [lhs_shape, rhs_shape, list(ag_full_host)],
            [torch_dtype_mm, torch_dtype_mm, torch_dtype_ag],
        )
        def main(
            lhs: Operand,
            rhs: Operand,
            ag_full: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            ctx = lhs.context
            f32 = F32Type.get(ctx)
            host_out_ty = RankedTensorType.get(out_shape, lhs.type.element_type)
            host_ag_full_ty = RankedTensorType.get(list(ag_full_host), f32)

            ag_streams = prepare_fabric_ring_all_gather_spatial_streams(
                builder,
                ag_full,
                per_device_shard_logical=ag_per_device_shard,
                gathered_mesh_logical=ag_gathered_on_mesh,
                mesh_shape=mesh_shape,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
                view_in_shape=ag_view_in_shape,
                view_out_shape=ag_view_out_shape,
                unit_attrs=unit_attrs,
            )

            lhs_metal_ty = builder.get_metal_tensor_layout(
                lhs_shape,
                grid=layout_cfg_r0["lhs_grid"],
                tiled=True,
                element_dtype=torch_dtype_mm,
            )
            rhs_metal_ty = builder.get_metal_tensor_layout(
                rhs_shape,
                grid=layout_cfg_r0["rhs_grid"],
                tiled=True,
                element_dtype=torch_dtype_mm,
            )
            out_metal_ty = builder.get_metal_tensor_layout(
                out_shape,
                grid=layout_cfg_r0["out_grid"],
                tiled=True,
                element_dtype=torch_dtype_mm,
            )

            r0_start = grid_ranges[0][0]
            r0_vg_inv, r0_vg_fwd = _build_virtual_grid_attrs(
                ctx, tensor_rank=len(lhs_metal_ty.shape), core_start=r0_start
            )

            lhs_m = prepare_metal_input(
                builder, lhs, lhs_metal_ty, r0_vg_inv, r0_vg_fwd, unit_attrs=unit_attrs
            )
            rhs_m = prepare_metal_input(
                builder, rhs, rhs_metal_ty, r0_vg_inv, r0_vg_fwd, unit_attrs=unit_attrs
            )
            out0_m = prepare_metal_output(out_metal_ty, r0_vg_inv, r0_vg_fwd)

            region_builders = [
                matmul_region_build(
                    builder,
                    lhs_m,
                    rhs_m,
                    out0_m,
                    out_block_shape=layout_cfg_r0["out_block_tiles"],
                ),
                all_gather_region_build(
                    builder,
                    ag_streams.view_in,
                    ag_streams.view_out,
                    ag_streams.start_sem,
                    ag_streams.end_sem,
                    num_mesh_devices=num_mesh_devices,
                ),
            ]

            spatial_results = builder.spatial(
                [lhs_m, rhs_m, ag_streams.view_in],
                [out0_m, ag_streams.view_out],
                grid_ranges,
                region_builders,
                result_types=[out0_m.type, ag_streams.view_out.type],
                unit_attrs=unit_attrs,
            )
            r0_m, r1_m = spatial_results[0], spatial_results[1]

            res_mm = builder.to_layout(
                r0_m, output_type=host_out_ty, unit_attrs=unit_attrs
            )

            res_ag = finalize_fabric_ring_all_gather_to_host(
                r1_m,
                gathered_mesh_logical=ag_gathered_on_mesh,
                host_full_ty=host_ag_full_ty,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            lhs_g = torch.randn(lhs_shape, dtype=torch_dtype_mm)
            rhs_g = torch.randn(rhs_shape, dtype=torch_dtype_mm)
            ag_full_g = torch.randn(list(ag_full_host), dtype=torch_dtype_ag)
            golden_mm = lhs_g @ rhs_g
            builder.set_goldens(
                {lhs: lhs_g, rhs: rhs_g, ag_full: ag_full_g},
                {res_mm: golden_mm, res_ag: ag_full_g},
            )
            return (res_mm, res_ag)

    compile_and_execute_d2m(
        spatial_module,
        target=target,
        device=device,
        mesh_dict=mesh_dict,
        pipeline_options=["mesh-topology=linear,ring"],
        custom_pipeline="ttir-to-ttmetal-pipeline{use-tile-matmul=false enable-l1-acc=true}",
        print_ir=False,
        check_pcc=False,
        **get_request_kwargs(request),
    )
