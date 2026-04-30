# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from builder.base.builder_enums import MeshShardDirection, MeshShardType
import pytest
import torch
from typing import Callable, List, Optional, OrderedDict, Tuple

from ttmlir.dialects import arith, d2m, tensor
from ttmlir.ir import (
    AffineConstantExpr,
    AffineDimExpr,
    AffineMap,
    AffineMapAttr,
    Context,
    DenseElementsAttr,
    IndexType,
    IntegerType,
    RankedTensorType,
    Type,
)

from builder.base.builder_utils import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m
from ttmlir.dialects import affine, arith, d2m, tensor, ttcore
from conftest import get_request_kwargs
from test_utils import (
    make_shard_shape,
)

pytestmark = pytest.mark.frontend("d2m")


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


def prepare_metal_input(
    builder: D2MBuilder,
    input_tensor: Operand,
    input_shape: List[int],
    core_start: Tuple[int, int],
) -> Operand:
    metal_type = builder.get_metal_tensor_layout(input_shape, tiled=True)
    (
        virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping,
    ) = _build_virtual_grid_attrs(
        input_tensor.context, tensor_rank=len(metal_type.shape), core_start=core_start
    )
    output = builder.empty(
        metal_type, virtual_grid_inverse_mapping, virtual_grid_forward_mapping
    )
    return builder.to_layout(input_tensor, output=output)


def prepare_metal_output(
    builder: D2MBuilder,
    out_shape: List[int],
    core_start: Tuple[int, int],
):
    out_metal_ty = builder.get_metal_tensor_layout(out_shape, tiled=True)
    (
        virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping,
    ) = _build_virtual_grid_attrs(
        builder.context, tensor_rank=len(out_metal_ty.shape), core_start=core_start
    )
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
    out_block_shape: List[int],
) -> Callable[[], None]:
    def _build():
        @builder.generic(
            grid=(1, 1),
            block_factors=(1, 1, 1),
            indexing_maps=(
                lambda m, n, k: (m, k),
                lambda m, n, k: (k, n),
                lambda m, n, k: (m, n),
            ),
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
    input: Operand,
    output: Operand,
    load_sem: Operand,
    store_sem: Operand,
) -> Callable[[], None]:
    def _build():
        ctx = builder.context
        c_wait = arith.constant(IndexType.get(ctx), 7)
        d0 = AffineDimExpr.get(0, ctx)
        d1 = AffineDimExpr.get(1, ctx)
        d2 = AffineDimExpr.get(2, ctx)
        map2 = AffineMap.get(3, 0, [d1], ctx)
        map3 = AffineMap.get(3, 0, [d0 + d2], ctx)

        @builder.generic(
            grid=(2, 1),
            block_factors=(),
            indexing_maps=(),
            iterator_types=[],
        )
        def ag_1x8(input, output, additional_args):
            _load_sem = additional_args[0]
            _store_sem = additional_args[1]
            mesh_row = d2m.mesh_position(dim=0)
            c1 = arith.constant(IndexType.get(ctx), 1)
            c0 = arith.constant(IndexType.get(ctx), 0)
            c8 = arith.constant(IndexType.get(ctx), 8)
            core0 = d2m.core_index(0)
            core1 = d2m.core_index(1)
            d2m.device_synchronize(
                _load_sem, [mesh_row, c0], [c1, c8], 7, [core0, core1]
            )
            core0_1 = d2m.core_index(0)
            loaded = builder.remote_load(input, [core0_1, c0])
            mesh_col = d2m.mesh_position(dim=1)
            idx_row = affine.apply(map2, [mesh_col, core0_1, c0])
            idx_col = affine.apply(map3, [mesh_col, core0_1, c0])
            stored = builder.remote_store(
                output.type,
                output,
                [idx_row, idx_col],
                start_device=[mesh_row, c0],
                device_mcast_shape=[c1, c8],
                semaphore_indices=[core0_1, c0],
                local_buffer=loaded,
                semaphore=_store_sem,
            )
            d2m.semaphore_wait(store_sem, c_wait)
            d2m.yield_([stored])

        d2m.spatial_yield(ag_1x8(input, output, additional_args=[load_sem, store_sem]))

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
    out_block_tiles = [lhs_shape[0] // 32, rhs_shape[1] // 32]
    torch_dtype = torch.float32

    def spatial_module(builder: D2MBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch_dtype, torch_dtype])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
        ):
            host_out_ty = RankedTensorType.get(out_shape, lhs.type.element_type)
            # Region r0.
            core_start_r0 = grid_ranges[0][0]
            lhs_m_r0 = prepare_metal_input(builder, lhs, lhs_shape, core_start_r0)
            rhs_m_r0 = prepare_metal_input(builder, rhs, rhs_shape, core_start_r0)
            out_m_r0 = prepare_metal_output(builder, out_shape, core_start_r0)

            # Region r1.
            core_start_r1 = grid_ranges[1][0]
            lhs_m_r1 = prepare_metal_input(builder, lhs, lhs_shape, core_start_r1)
            rhs_m_r1 = prepare_metal_input(builder, rhs, rhs_shape, core_start_r1)
            out_m_r1 = prepare_metal_output(builder, out_shape, core_start_r1)

            region_builders = [
                matmul_region_build(
                    builder,
                    lhs_m_r0,
                    rhs_m_r0,
                    out_m_r0,
                    out_block_shape=out_block_tiles,
                ),
                matmul_region_build(
                    builder,
                    lhs_m_r1,
                    rhs_m_r1,
                    out_m_r1,
                    out_block_shape=out_block_tiles,
                ),
            ]

            spatial_results = builder.spatial(
                [lhs_m_r0, rhs_m_r0, lhs_m_r1, rhs_m_r1],
                [out_m_r0, out_m_r1],
                grid_ranges,
                region_builders,
                result_types=[out_m_r0.type, out_m_r1.type],
            )
            result_m_r0, result_m_r1 = spatial_results[0], spatial_results[1]

            res_r0 = builder.to_layout(result_m_r0, output_type=host_out_ty)
            res_r1 = builder.to_layout(result_m_r1, output_type=host_out_ty)

            lhs_g = torch.randn(lhs_shape, dtype=torch_dtype)
            rhs_g = torch.randn(rhs_shape, dtype=torch_dtype)
            golden = lhs_g @ rhs_g
            builder.set_goldens(
                {lhs: lhs_g, rhs: rhs_g},
                {res_r0: golden, res_r1: golden},
            )
            return (res_r0, res_r1)

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
    out_block_tiles = [lhs_shape[0] // 32, rhs_shape[1] // 32]

    torch_dtype = torch.bfloat16

    def spatial_module(builder: D2MBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch_dtype, torch_dtype])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
        ):
            host_out_ty = RankedTensorType.get(out_shape, lhs.type.element_type)
            core_start = grid_range_single_11[0][0]
            lhs_m = prepare_metal_input(builder, lhs, lhs_shape, core_start)
            rhs_m = prepare_metal_input(builder, rhs, rhs_shape, core_start)
            out_m = prepare_metal_output(builder, out_shape, core_start)

            r_m = builder.spatial(
                [lhs_m, rhs_m],
                [out_m],
                grid_range_single_11,
                [
                    matmul_region_build(
                        builder, lhs_m, rhs_m, out_m, out_block_shape=out_block_tiles
                    )
                ],
                result_types=[out_m.type],
            )

            res = builder.to_layout(r_m, output_type=host_out_ty)

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


def _global_semaphore_backing_tensor_type(ctx: Context) -> RankedTensorType:
    """8x8x1x1 ui32 L1 sharded backing tensor for create_global_semaphore."""

    i64 = IntegerType.get_signless(64)
    collapse_ty = RankedTensorType.get([2, 2], i64)
    collapse = DenseElementsAttr.get([[0, 1], [1, 2]], type=collapse_ty)
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


@pytest.mark.parametrize(
    "mesh_shape",
    [
        pytest.param((1, 8), id="1x1"),
    ],
)
@pytest.mark.parametrize(
    "test_shape",
    [
        pytest.param((32, 32), id="32x32"),
    ],
)
def test_single_allgather(
    mesh_shape: Tuple[int, int],
    test_shape: Tuple[int, int],
    request,
    device,
):
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: D2MBuilder):
        @builder.func([full_input_shape], [torch.float32])
        def all_gather(input: Operand, builder: D2MBuilder):
            in_shard = builder.mesh_shard(
                input,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

            in_tile = prepare_metal_input(builder, in_shard, test_shape, (0, 0))
            out_local_shape = [
                test_shape[0] // mesh_shape[0],
                test_shape[1] // mesh_shape[1],
            ]
            out_tile = prepare_metal_output(builder, out_local_shape, (0, 0))

            # TODO: Use the correct output type
            out_shard = builder.empty(in_shard.type)

            sem_ty = _global_semaphore_backing_tensor_type(input.context)
            sem_gs_ty = Type.parse("!d2m.global_semaphore", input.context)
            load_sem = d2m.create_global_semaphore(
                d2m.empty(sem_ty), value=0, results=[sem_gs_ty]
            )
            store_sem = d2m.create_global_semaphore(
                d2m.empty(sem_ty), value=0, results=[sem_gs_ty]
            )
            region_builders = [
                all_gather_region_build(builder, in_tile, out_tile, load_sem, store_sem)
            ]
            spatial_results = builder.spatial(
                [in_shard],
                [out_shard],
                [((0, 0), (8, 8))],
                region_builders,
                result_types=[out_shard.type],
            )
            out_tensor = builder.mesh_shard(
                spatial_results[0],
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
            return out_tensor

    pipeline_options = [
        f"mesh-topology=linear,ring",
    ]

    compile_and_execute_d2m(
        module,
        target="ttmetal",
        device=device,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=pipeline_options,
        **get_request_kwargs(request),
    )
