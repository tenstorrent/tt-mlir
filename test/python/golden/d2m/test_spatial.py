# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from builder.base.builder_enums import MeshShardDirection, MeshShardType
import pytest
import torch
from typing import Callable, List, Tuple

import _ttmlir_runtime as tt_runtime
from ttmlir.ir import (
    Attribute,
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
from ttmlir.dialects import affine, arith, d2m, tensor, ttcore, ttir, ttnn
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("d2m")


def _build_virtual_grid_attrs(
    ctx, tensor_rank: int, core_start: Tuple[int, int]
) -> Tuple[AffineMapAttr, AffineMapAttr]:
    assert tensor_rank >= 2, f"Expected tensor rank >= 2, got {tensor_rank}"

    start_y, start_x = core_start
    d0 = AffineDimExpr.get(0, ctx)
    d1 = AffineDimExpr.get(1, ctx)
    c0 = AffineConstantExpr.get(0, ctx)
    c_start_y = AffineConstantExpr.get(start_y, ctx)
    c_start_x = AffineConstantExpr.get(start_x, ctx)

    # Offset only the virtual-grid axes; pass remaining axes through unchanged.
    vgm_fwd_results = [d0 + c_start_y, d1 + c_start_x]
    for i in range(2, tensor_rank):
        vgm_fwd_results.append(AffineDimExpr.get(i, ctx))
    vgm_fwd = AffineMap.get(tensor_rank, 0, vgm_fwd_results, ctx)

    # Physical -> virtual mapping for 2D core coordinates.
    vgm_inv = AffineMap.get(2, 0, [c0, d0 - c_start_y, d1 - c_start_x], ctx)
    return AffineMapAttr.get(vgm_inv), AffineMapAttr.get(vgm_fwd)


def _build_ttnn_layout_attr(
    builder: D2MBuilder,
    logical_shape: List[int],
    element_type: Type,
    *,
    buffer_type: ttnn.BufferType,
    tensor_memory_layout: ttnn.TensorMemoryLayout,
    core_range: Tuple[Tuple[int, int], Tuple[int, int]] | None = None,
) -> Attribute:
    core_range_set = None
    if tensor_memory_layout == ttnn.TensorMemoryLayout.BlockSharded:
        if core_range is None:
            raise ValueError("core_range is required for block_sharded layout")
        start, end = core_range
        core_range_set = ttnn.ir.CoreRangeSetAttr.get(
            builder.context,
            [
                ttnn.ir.CoreRangeAttr.get(
                    builder.context,
                    ttnn.ir.CoreCoordAttr.get(builder.context, *start),
                    ttnn.ir.CoreCoordAttr.get(builder.context, *end),
                )
            ],
        )

    return builder._create_ttnn_tensor_encoding(
        shape=logical_shape,
        element_type=element_type,
        buffer_type=buffer_type,
        tensor_memory_layout=tensor_memory_layout,
        grid_shape=[1, 1],
        core_range_set=core_range_set,
    )


def _core_range_start_and_shape(
    core_range: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    (start_y, start_x), (end_y, end_x) = core_range
    grid_shape = (end_y - start_y + 1, end_x - start_x + 1)
    return (start_y, start_x), grid_shape


def _largest_divisor_not_exceeding(value: int, upper_bound: int) -> int:
    if value <= 0:
        return 1
    capped = min(value, upper_bound)
    for divisor in range(capped, 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


def _resolve_effective_grid(
    logical_shape: List[int], requested_grid: Tuple[int, int]
) -> Tuple[int, int]:
    tile_h = logical_shape[0] // 32
    tile_w = logical_shape[1] // 32
    return (
        _largest_divisor_not_exceeding(tile_h, requested_grid[0]),
        _largest_divisor_not_exceeding(tile_w, requested_grid[1]),
    )


def _convert_input_to_device_tiled_ttmetal(
    builder: D2MBuilder,
    input_tensor: Operand,
    input_shape: List[int],
    core_start: Tuple[int, int],
    grid_shape: Tuple[int, int] = (1, 1),
) -> Operand:
    metal_type = builder.get_metal_tensor_layout(
        input_shape, tiled=True, grid=grid_shape
    )
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


def _convert_input_to_device_tiled_ttnn(
    builder: D2MBuilder,
    input_tensor: Operand,
    input_shape: List[int],
    core_start: Tuple[int, int],
    grid_shape: Tuple[int, int],
) -> Operand:
    ttnn_in_ty = RankedTensorType.get(
        input_shape,
        input_tensor.type.element_type,
        _build_ttnn_layout_attr(
            builder,
            input_shape,
            input_tensor.type.element_type,
            buffer_type=ttnn.BufferType.DRAM,
            tensor_memory_layout=ttnn.TensorMemoryLayout.Interleaved,
        ),
    )
    ttnn_input = ttir.to_layout([ttnn_in_ty], input_tensor, builder.empty(ttnn_in_ty))
    metal_type = builder.get_metal_tensor_layout(
        input_shape, tiled=True, grid=grid_shape
    )
    (
        virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping,
    ) = _build_virtual_grid_attrs(
        input_tensor.context, tensor_rank=len(metal_type.shape), core_start=core_start
    )
    return ttir.ttnn_metal_layout_cast(
        metal_type,
        ttnn_input,
        virtual_grid_inverse_mapping=virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping=virtual_grid_forward_mapping,
    )


def _convert_input_to_device_tiled(
    builder: D2MBuilder,
    tensor: Operand,
    tensor_shape: List[int],
    core_range: Tuple[Tuple[int, int], Tuple[int, int]],
    target: str,
    grid_shape: Tuple[int, int] | None = None,
) -> Operand:
    core_start, core_range_shape = _core_range_start_and_shape(core_range)
    if grid_shape is None:
        grid_shape = core_range_shape
    effective_grid = _resolve_effective_grid(tensor_shape, grid_shape)
    if target == "ttnn-mode":
        return _convert_input_to_device_tiled_ttnn(
            builder, tensor, tensor_shape, core_start, grid_shape=effective_grid
        )
    if target == "ttmetal":
        return _convert_input_to_device_tiled_ttmetal(
            builder, tensor, tensor_shape, core_start, grid_shape=effective_grid
        )
    raise ValueError(f"Unsupported target: {target}")


def _allocate_device_output_ttmetal(
    builder: D2MBuilder,
    out_shape: List[int],
    output_element_type: Type,
    element_dtype: torch.dtype,
    core_start: Tuple[int, int],
    grid_shape: Tuple[int, int] = (1, 1),
) -> Tuple[Operand, RankedTensorType]:
    out_metal_ty = builder.get_metal_tensor_layout(
        out_shape, tiled=True, element_dtype=element_dtype, grid=grid_shape
    )
    (
        virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping,
    ) = _build_virtual_grid_attrs(
        builder.context, tensor_rank=len(out_metal_ty.shape), core_start=core_start
    )
    result_output_ty = RankedTensorType.get(out_shape, output_element_type)
    return (
        builder.empty(
            out_metal_ty,
            virtual_grid_inverse_mapping=virtual_grid_inverse_mapping,
            virtual_grid_forward_mapping=virtual_grid_forward_mapping,
        ),
        result_output_ty,
    )


def _allocate_device_output_ttnn(
    builder: D2MBuilder,
    out_shape: List[int],
    output_element_type: Type,
    element_dtype: torch.dtype,
    core_start: Tuple[int, int],
    grid_shape: Tuple[int, int],
) -> Tuple[Operand, RankedTensorType]:
    end_y = core_start[0] + grid_shape[0] - 1
    end_x = core_start[1] + grid_shape[1] - 1
    effective_core_range = (core_start, (end_y, end_x))
    ttnn_out_ty = RankedTensorType.get(
        out_shape,
        output_element_type,
        _build_ttnn_layout_attr(
            builder,
            out_shape,
            output_element_type,
            buffer_type=ttnn.BufferType.L1,
            tensor_memory_layout=ttnn.TensorMemoryLayout.BlockSharded,
            core_range=effective_core_range,
        ),
    )
    out_ttnn = builder.empty(ttnn_out_ty)
    out_metal_ty = builder.get_metal_tensor_layout(
        out_shape, tiled=True, element_dtype=element_dtype, grid=grid_shape
    )
    (
        virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping,
    ) = _build_virtual_grid_attrs(
        builder.context, tensor_rank=len(out_metal_ty.shape), core_start=core_start
    )
    out_metal = ttir.ttnn_metal_layout_cast(
        out_metal_ty,
        out_ttnn,
        virtual_grid_inverse_mapping=virtual_grid_inverse_mapping,
        virtual_grid_forward_mapping=virtual_grid_forward_mapping,
    )
    return out_metal, ttnn_out_ty


def _allocate_device_output(
    builder: D2MBuilder,
    tensor_shape: List[int],
    output_element_type: Type,
    element_dtype: torch.dtype,
    core_range: Tuple[Tuple[int, int], Tuple[int, int]],
    target: str,
    grid_shape: Tuple[int, int] | None = None,
) -> Tuple[Operand, RankedTensorType]:
    core_start, core_range_shape = _core_range_start_and_shape(core_range)
    if grid_shape is None:
        grid_shape = core_range_shape
    effective_grid = _resolve_effective_grid(tensor_shape, grid_shape)
    if target == "ttnn-mode":
        return _allocate_device_output_ttnn(
            builder,
            tensor_shape,
            output_element_type,
            element_dtype,
            core_start,
            grid_shape=effective_grid,
        )
    if target == "ttmetal":
        return _allocate_device_output_ttmetal(
            builder,
            tensor_shape,
            output_element_type,
            element_dtype,
            core_start,
            grid_shape=effective_grid,
        )
    raise ValueError(f"Unsupported target: {target}")


def _convert_result_to_host_ttnn(
    output_metal: Operand, result_output_ty: RankedTensorType
):
    return ttir.ttnn_metal_layout_cast(result_output_ty, output_metal)


def _convert_result_to_host_ttmetal(
    builder: D2MBuilder, output_metal: Operand, result_output_ty: RankedTensorType
):
    return builder.to_layout(output_metal, output_type=result_output_ty)


def _convert_result_to_host(
    builder: D2MBuilder,
    output_metal: Operand,
    target: str,
    result_output_ty: RankedTensorType | None = None,
):
    if result_output_ty is None:
        raise ValueError("result_output_ty is required")
    if target == "ttnn-mode":
        return _convert_result_to_host_ttnn(output_metal, result_output_ty)
    if target == "ttmetal":
        return _convert_result_to_host_ttmetal(builder, output_metal, result_output_ty)
    raise ValueError(f"Unsupported target: {target}")


def _create_global_semaphore(
    builder: D2MBuilder,
):
    # Global semaphore allocation is currently forced to use the full device
    # grid to increase reuse probability and reduce integration complexity.
    sem_grid_shape = [8, 8]
    sem_backing_type = builder.get_metal_tensor_layout(
        sem_grid_shape,
        tiled=False,
        element_dtype=torch.uint32,
        grid=tuple(sem_grid_shape),
        dim_alignments=(1, 1),
    )
    return builder.create_global_semaphore(output_type=sem_backing_type, value=0)


def _build_matmul_region(
    builder: D2MBuilder,
    lhs: Operand,
    rhs: Operand,
    out: Operand,
    out_block_shape: List[int],
    region_grid: Tuple[int, int] = (1, 1),
) -> Callable[[], None]:
    def _build():
        @builder.generic(
            grid=region_grid,
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


def _build_all_gather_region(
    builder: D2MBuilder,
    input: Operand,
    output: Operand,
    load_sem: Operand,
    store_sem: Operand,
) -> Callable[[], None]:
    def _build():
        ctx = builder.context
        fabric_connection_config = Attribute.parse(
            "#ttcore.fabric_connection_config<noc_index = noc0, topology = ring, cluster_axis = 1, routing_mode = unidir_ring_torus, num_links = 1>",
            ctx,
        )
        d0 = AffineDimExpr.get(0, ctx)
        d1 = AffineDimExpr.get(1, ctx)
        d2 = AffineDimExpr.get(2, ctx)
        map2 = AffineMap.get(3, 0, [d1], ctx)
        map3 = AffineMap.get(3, 0, [d0 + d2], ctx)

        @builder.generic(
            # dynamic grid selection with allgather is not supported so far, so we hardcode the grid to (2, 1).
            # see TTIRToD2M conversion pass code for more details.
            # TODO: update this test case once dynamic grid selection is supported.
            grid=(2, 1),
            block_factors=(),
            indexing_maps=(),
            iterator_types=[],
            fabric_connection_config=fabric_connection_config,
        )
        def ag_1x8(input, output, load_sem, store_sem):
            mesh_row = d2m.mesh_position(dim=0)
            c1 = arith.constant(IndexType.get(ctx), 1)
            c0 = arith.constant(IndexType.get(ctx), 0)
            c8 = arith.constant(IndexType.get(ctx), 8)
            c_wait = arith.constant(IndexType.get(ctx), 7)
            core0 = d2m.core_index(0)
            core1 = d2m.core_index(1)
            d2m.device_synchronize(
                load_sem, [mesh_row, c0], [c1, c8], 7, [core0, core1]
            )
            core0_1 = d2m.core_index(0)
            loaded = builder.remote_load(input, [core0_1, c0])
            mesh_col = d2m.mesh_position(dim=1)
            idx_row = affine.apply(map2, [mesh_col, core0_1, c0])
            idx_col = affine.apply(map3, [mesh_col, core0_1, c0])
            stored = d2m.remote_store(
                output.type,
                output,
                [idx_row, idx_col],
                start_device=[mesh_row, c0],
                device_mcast_shape=[c1, c8],
                semaphore_indices=[core0_1, c0],
                local_buffer=loaded,
                semaphore=store_sem,
            )
            d2m.semaphore_wait(store_sem, c_wait)
            d2m.yield_([stored])

        d2m.spatial_yield(
            [
                ag_1x8(
                    input,
                    output,
                    additional_args=[load_sem, store_sem],
                )
            ]
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
        pytest.param("ttmetal", id="ttmetal"),
        pytest.param("ttnn-mode", id="ttnn-mode"),
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
    torch_dtype = torch.float32
    # builder create module input with ttnn layout encoding if ttnn_inputs is True
    ttnn_inputs = target == "ttnn-mode"

    def spatial_module(builder: D2MBuilder):
        @builder.func(
            [lhs_shape, rhs_shape],
            [torch_dtype, torch_dtype],
            ttnn_inputs=ttnn_inputs,
        )
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
        ):
            core_range_r0 = grid_ranges[0]
            lhs_m_r0 = _convert_input_to_device_tiled(
                builder, lhs, lhs_shape, core_range_r0, target, grid_shape=(1, 1)
            )
            rhs_m_r0 = _convert_input_to_device_tiled(
                builder,
                rhs,
                rhs_shape,
                core_range_r0,
                target,
            )
            out_m_r0, out_result_ty_r0 = _allocate_device_output(
                builder,
                out_shape,
                lhs.type.element_type,
                torch_dtype,
                core_range_r0,
                target,
            )
            region_grid_r0 = tuple(out_m_r0.type.shape[:2])
            out_block_shape_r0 = list(out_m_r0.type.shape[-2:])

            core_range_r1 = grid_ranges[1]
            lhs_m_r1 = _convert_input_to_device_tiled(
                builder, lhs, lhs_shape, core_range_r1, target, grid_shape=(1, 1)
            )
            rhs_m_r1 = _convert_input_to_device_tiled(
                builder,
                rhs,
                rhs_shape,
                core_range_r1,
                target,
            )
            out_m_r1, out_result_ty_r1 = _allocate_device_output(
                builder,
                out_shape,
                lhs.type.element_type,
                torch_dtype,
                core_range_r1,
                target,
            )
            region_grid_r1 = tuple(out_m_r1.type.shape[:2])
            out_block_shape_r1 = list(out_m_r1.type.shape[-2:])

            region_builders = [
                _build_matmul_region(
                    builder,
                    lhs_m_r0,
                    rhs_m_r0,
                    out_m_r0,
                    out_block_shape=out_block_shape_r0,
                    region_grid=region_grid_r0,
                ),
                _build_matmul_region(
                    builder,
                    lhs_m_r1,
                    rhs_m_r1,
                    out_m_r1,
                    out_block_shape=out_block_shape_r1,
                    region_grid=region_grid_r1,
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

            res_r0 = _convert_result_to_host(
                builder,
                result_m_r0,
                target,
                result_output_ty=out_result_ty_r0,
            )
            res_r1 = _convert_result_to_host(
                builder,
                result_m_r1,
                target,
                result_output_ty=out_result_ty_r1,
            )

            lhs_g = torch.randn(lhs_shape, dtype=torch_dtype)
            rhs_g = torch.randn(rhs_shape, dtype=torch_dtype)
            golden = lhs_g @ rhs_g
            builder.set_goldens(
                {lhs: lhs_g, rhs: rhs_g},
                {res_r0: golden, res_r1: golden},
            )
            return (res_r0, res_r1)

    # Temporary: disable PCC check for ttnn-mode target due to known mismatch.
    check_pcc = False if target == "ttnn-mode" else True
    compile_and_execute_d2m(
        spatial_module,
        target=target,
        device=device,
        check_pcc=check_pcc,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize(
    "grid_ranges",
    [
        pytest.param([((0, 0), (1, 0)), ((0, 1), (0, 1))], id="offset_x"),
        pytest.param([((0, 0), (1, 0)), ((1, 1), (1, 1))], id="none_from_origin"),
        pytest.param([((0, 0), (1, 0)), ((2, 2), (2, 3))], id="multi_cores"),
    ],
)
@pytest.mark.parametrize(
    "mesh_shape",
    [
        pytest.param((1, 8), id="1x8"),
    ],
)
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("ttmetal", id="ttmetal"),
        pytest.param(
            "ttnn-mode",
            id="ttnn-mode",
            marks=pytest.mark.skip(
                reason="Temporarily skip ttnn-mode target for all-gather spatial path."
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING]
)
@pytest.mark.parametrize(
    "ag_test_shape",
    [
        pytest.param((256, 256), id="256x256"),
    ],
)
@pytest.mark.parametrize(
    "mm_shapes",
    [
        pytest.param(([32, 32], [32, 32]), id="32x32_32x32"),
        pytest.param(([256, 256], [256, 256]), id="256x256_256x256"),
    ],
)
def test_spatial_two_regions_allgather_and_matmul(
    target: str,
    mesh_shape: Tuple[int, int],
    fabric_config: tt_runtime.runtime.FabricConfig,
    ag_test_shape: Tuple[int, int],
    mm_shapes: Tuple[List[int], List[int]],
    request,
    device,
    grid_ranges,
):
    mm_lhs_shape, mm_rhs_shape = mm_shapes
    ag_input_shape = [ag_test_shape[0], ag_test_shape[1] * mesh_shape[1]]
    ag_input_rank = len(ag_test_shape)
    mm_out_shape = [mm_lhs_shape[0], mm_rhs_shape[1]]
    # builder create module input with ttnn layout encoding if ttnn_inputs is True
    ttnn_inputs = target == "ttnn-mode"

    def spatial_module(builder: D2MBuilder):
        @builder.func(
            [ag_input_shape, mm_lhs_shape, mm_rhs_shape],
            [torch.float32, torch.float32, torch.float32],
            ttnn_inputs=ttnn_inputs,
        )
        def main(
            ag_input: Operand,
            mm_lhs: Operand,
            mm_rhs: Operand,
            builder: D2MBuilder,
        ):
            # Region r0: all-gather op setup.
            ag_core_range = grid_ranges[0]
            ag_mesh_sharded_input = builder.mesh_shard(
                ag_input,
                MeshShardType.Devices,
                MeshShardDirection.FullToShard,
                mesh_shape,
                [-1, 1],
            )
            ag_in_m = _convert_input_to_device_tiled(
                builder,
                ag_mesh_sharded_input,
                list(ag_mesh_sharded_input.type.shape),
                ag_core_range,
                target,
                grid_shape=(ag_input_rank, 1),
            )
            ag_out_local_shape = ag_input_shape
            ag_out_m, _ = _allocate_device_output(
                builder,
                ag_out_local_shape,
                ag_input.type.element_type,
                torch.float32,
                ag_core_range,
                target,
                grid_shape=(ag_input_rank, mesh_shape[1]),
            )

            # Region r1: matmul op setup.
            mm_core_range = grid_ranges[1]
            mm_mesh_sharded_lhs = builder.mesh_shard(
                mm_lhs,
                MeshShardType.Replicate,
                MeshShardDirection.FullToShard,
                mesh_shape,
                [-1],
            )
            mm_mesh_sharded_rhs = builder.mesh_shard(
                mm_rhs,
                MeshShardType.Replicate,
                MeshShardDirection.FullToShard,
                mesh_shape,
                [-1],
            )
            mm_lhs_m = _convert_input_to_device_tiled(
                builder,
                mm_mesh_sharded_lhs,
                list(mm_mesh_sharded_lhs.type.shape),
                mm_core_range,
                target,
                grid_shape=(1, 1),
            )
            mm_rhs_m = _convert_input_to_device_tiled(
                builder,
                mm_mesh_sharded_rhs,
                list(mm_mesh_sharded_rhs.type.shape),
                mm_core_range,
                target,
            )
            mm_out_m, _ = _allocate_device_output(
                builder,
                mm_out_shape,
                mm_lhs.type.element_type,
                torch.float32,
                mm_core_range,
                target,
            )
            _mm_region_grid = tuple(mm_out_m.type.shape[:2])
            mm_region_out_block_tiles = list(mm_out_m.type.shape[-2:])

            # All-gather synchronization resources.
            load_sem = _create_global_semaphore(builder)
            store_sem = _create_global_semaphore(builder)

            # Build and execute spatial regions.
            region_builders = [
                _build_all_gather_region(
                    builder,
                    ag_in_m,
                    ag_out_m,
                    load_sem,
                    store_sem,
                ),
                _build_matmul_region(
                    builder,
                    mm_lhs_m,
                    mm_rhs_m,
                    mm_out_m,
                    out_block_shape=mm_region_out_block_tiles,
                    region_grid=_mm_region_grid,
                ),
            ]

            spatial_results = builder.spatial(
                [ag_in_m, mm_lhs_m, mm_rhs_m],
                [ag_out_m, mm_out_m],
                grid_ranges,
                region_builders,
                result_types=[ag_out_m.type, mm_out_m.type],
            )
            ag_result_m, mm_result_m = spatial_results[0], spatial_results[1]

            # Region r0: all-gather result materialization.
            ag_mesh_out_ty = RankedTensorType.get(
                [ag_out_local_shape[0], ag_out_local_shape[1]],
                Type.parse("f32", builder.context),
                Attribute.parse('#ttcore.tensor_mesh<"mesh">', builder.context),
            )
            ag_mesh_out_tensor = builder.to_layout(
                ag_result_m, output_type=ag_mesh_out_ty
            )
            ag_result = builder.mesh_shard(
                ag_mesh_out_tensor,
                MeshShardType.Devices,
                MeshShardDirection.ShardToFull,
                mesh_shape,
                [0, 1],
            )

            # Region r1: matmul result materialization.
            mm_mesh_out_ty = RankedTensorType.get(
                [mm_out_shape[0], mm_out_shape[1]],
                Type.parse("f32", builder.context),
                Attribute.parse('#ttcore.tensor_mesh<"mesh">', builder.context),
            )
            mm_mesh_out_tensor = builder.to_layout(
                mm_result_m, output_type=mm_mesh_out_ty
            )
            mm_result = builder.mesh_shard(
                mm_mesh_out_tensor,
                MeshShardType.Replicate,
                MeshShardDirection.ShardToFull,
                mesh_shape,
                [-1],
            )

            # Goldens for all-gather and matmul outputs.
            ag_input_golden = torch.randn(ag_input_shape, dtype=torch.float32)
            mm_lhs_golden = torch.randn(mm_lhs_shape, dtype=torch.float32)
            mm_rhs_golden = torch.randn(mm_rhs_shape, dtype=torch.float32)
            ag_golden = torch.cat(
                [ag_input_golden for _ in range(mesh_shape[1])], dim=1
            )
            mm_golden = mm_lhs_golden @ mm_rhs_golden
            builder.set_goldens(
                {
                    ag_input: ag_input_golden,
                    mm_lhs: mm_lhs_golden,
                    mm_rhs: mm_rhs_golden,
                },
                {ag_result: ag_golden, mm_result: mm_golden},
            )
            return (ag_result, mm_result)

    # Temporary: disable PCC check for ttnn-mode target due to known mismatch.
    check_pcc = False if target == "ttnn-mode" else True
    compile_and_execute_d2m(
        spatial_module,
        target=target,
        device=device,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=["mesh-topology=linear,ring"],
        check_pcc=check_pcc,
        **get_request_kwargs(request),
    )
