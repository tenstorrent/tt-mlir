# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

from ttmlir.ir import *
from ttmlir.dialects import ttcore

ctx = Context()


def createTensorLayout(
    shape,
    grid,
    memorySpace=ttcore.MemorySpace.DeviceL1,
    collapseIntervals=[(0, -1)],
    oobVal=ttcore.OOBVal.Undef,
):
    if isinstance(grid, list) or isinstance(grid, tuple):
        grid = ttcore.ir.GridAttr.get(ctx, list(grid))
    tensorTy = RankedTensorType.get(
        shape, F32Type.get(ctx), None, Location.unknown(ctx)
    )
    layout = tt.ir.MetalLayoutAttr.get(ctx, shape, grid, collapseIntervals, oobVal)
    return RankedTensorType.get(shape, F32Type.get(ctx), layout, Location.unknown(ctx))


def tilize(tensor, dataType, tileShape=[32, 32]):
    assert len(tileShape) == 2
    return ttcore.ir.MetalLayoutAttr.with_element_type_(
        ctx,
        tensor.encoding,
        ttcore.ir.TileType.get(ctx, tileShape[0], tileShape[1], dataType),
    )


def parallelize(tensor, grid, collapseIntervals=[(0, -1)]):
    if isinstance(grid, list) or isinstance(grid, tuple):
        grid = tt.ir.GridAttr.get(ctx, list(grid))
    return tt.ir.MetalLayoutAttr.get(
        ctx, tensor.encoding, tensor.shape, grid, collapseIntervals
    )


t0 = createTensorLayout([2, 3, 64, 128], [2, 4])
# CHECK: tensor<2x4x96x32xf32, #tt.metal_layout<[384, 128], undef, l1, dim_alignments = [32, 32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>>
print(t0)
# CHECK: #tt.metal_layout<[384, 128], undef, l1, dim_alignments = [32, 32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>
print(tilize(t0, tt.DataType.BFP_BFloat8).wrapped())
print(parallelize(t0, [3, 2]).wrapped())

t1 = createTensorLayout([2, 3, 64, 128], [2, 2, 4], collapseIntervals=[(1, -1)])
print(tilize(t1, ttcore.DataType.BFP_BFloat8).wrapped())
print(parallelize(t1, [3, 2]).wrapped())

t2 = createTensorLayout([128], [4], collapseIntervals=[(0, -1)])
# CHECK: tensor<4x32xf32, #tt.metal_layout<[128], undef, l1, dim_alignments = [32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>>
print(t2)
# CHECK: #tt.metal_layout<[128], undef, l1, dim_alignments = [32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>
print(parallelize(t2, [2]).wrapped())
# CHECK: #tt.metal_layout<[128], undef, l1, dim_alignments = [32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>
print(parallelize(t2, [1, 2]).wrapped())

t3 = createTensorLayout([128], [1, 4], collapseIntervals=[(0, -1)])
# CHECK: tensor<1x4x32xf32, #tt.metal_layout<[128], undef, l1, dim_alignments = [32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>>
print(t3)
# CHECK: #tt.metal_layout<[128], undef, l1, dim_alignments = [32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>
print(tilize(t3, tt.DataType.BFP_BFloat8).wrapped())

t4 = createTensorLayout([128], [1, 2, 4], collapseIntervals=[(0, -1)])
# CHECK: tensor<1x2x4x32xf32, #tt.metal_layout<[128], undef, l1, dim_alignments = [32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>>
print(t4)

# CHECK: #tt.metal_layout<[128], undef, l1, dim_alignments = [32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>
print(tilize(t4, tt.DataType.BFP_BFloat8).wrapped())
# CHECK: #tt.metal_layout<[128], undef, l1, dim_alignments = [32], collapse_dims = dense<[[0, -1]]> : tensor<1x2xi64>>
print(parallelize(t4, [1, 2]).wrapped())
