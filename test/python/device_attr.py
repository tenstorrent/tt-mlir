# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

from ttmlir.ir import *
from ttmlir.dialects import ttcore

ctx = Context()


def updiv(n, d):
    return (n + d - 1) // d


def volume(shape):
    vol = 1
    for dim in shape:
        vol *= dim
    return vol


def inferVirtToPhysicalMap(grid, physicalGrid=[8, 8], mesh=[1]):
    assert len(grid) >= 2
    mesh = mesh.copy()
    while len(mesh) < len(grid):
        mesh.append(1)
    totalDevices = volume(mesh)
    p0 = AffineConstantExpr.get(physicalGrid[0], ctx)
    p1 = AffineConstantExpr.get(physicalGrid[1], ctx)
    dZ = AffineConstantExpr.get(0, ctx)
    dY = AffineDimExpr.get(len(grid) - 2, ctx)
    dX = AffineDimExpr.get(len(grid) - 1, ctx)
    if totalDevices > 1:
        v = mesh[-1]
        if mesh[-1] > 1:
            dZ = dZ + AffineFloorDivExpr.get(dX, p1)
        if mesh[-2] > 1:
            dZ = dZ + AffineFloorDivExpr.get(dY, p0) * v
        for d in range(len(mesh) - 3, -1, -1):
            v *= mesh[d + 1]
            dn = AffineDimExpr.get(d, ctx)
            dZ = dn * v + dZ
    exprs = [
        dZ,
        dY % p0 if mesh[-2] > 1 else dY,
        dX % p1 if mesh[-1] > 1 else dX,
    ]
    return AffineMap.get(len(grid), 0, exprs, ctx)


def inferPhysicalToVirtMap(grid, physicalGrid=[8, 8], mesh=[1]):
    """
    Creates the inverse mapping from physical coordinates back to virtual coordinates.

    Physical input: (device_id, physical_y, physical_x) - always 3 dims
    Virtual output: (virtual_dim_0, virtual_dim_1, ..., virtual_dim_n-1) - len(grid) dims

    Algorithm:
    1. Expand device_id into mesh coordinates using floordiv/mod
       e.g., device 7 on mesh (2, 4, 2) -> mesh coord (0, 3, 1)
    2. For the last 2 mesh dimensions, multiply by physicalGrid and add physical coords
       virtual_y = mesh_coord[-2] * physicalGrid[0] + physical_y
       virtual_x = mesh_coord[-1] * physicalGrid[1] + physical_x
    3. For any leading mesh dimensions (batch dims), pass through directly
    """
    assert len(grid) >= 2
    mesh = mesh.copy()
    while len(mesh) < len(grid):
        mesh.append(1)

    # Physical dimensions: d0 = device_id, d1 = physical_y, d2 = physical_x
    numPhysicalDims = 3
    deviceId = AffineDimExpr.get(0, ctx)
    physY = AffineDimExpr.get(1, ctx)
    physX = AffineDimExpr.get(2, ctx)
    # Physical grid dimensions: p0 = physical_y, p1 = physical_x
    p0 = AffineConstantExpr.get(physicalGrid[0], ctx)
    p1 = AffineConstantExpr.get(physicalGrid[1], ctx)

    # Step 1: Expand device_id into mesh coordinates
    # Device ID is linearized as: sum over i of (mesh_coord[i] * product of mesh[i+1:])
    # We recover mesh_coord[i] = (deviceId / stride) % mesh[i]
    meshCoords = []
    for i in range(len(mesh)):
        # stride = product of mesh[i+1:]
        stride = 1
        for j in range(i + 1, len(mesh)):
            stride *= mesh[j]
        strideExpr = AffineConstantExpr.get(stride, ctx)
        meshDimSize = AffineConstantExpr.get(mesh[i], ctx)

        if stride > 1:
            coord = AffineFloorDivExpr.get(deviceId, strideExpr) % meshDimSize
        else:
            coord = deviceId % meshDimSize
        meshCoords.append(coord)

    # Step 2: Build virtual coordinates
    #   for leading dimensions (batch dims) - just the mesh coordinates
    #   for last 2 dimensions - multiply by physicalGrid and add physical coords
    exprs = []
    for i in range(len(mesh)):
        if i == len(mesh) - 2:
            exprs.append(meshCoords[-2] * p0 + physY)
        elif i == len(mesh) - 1:
            exprs.append(meshCoords[-1] * p1 + physX)
        else:
            exprs.append(meshCoords[i])

    return AffineMap.get(numPhysicalDims, 0, exprs, ctx)


def inferMemoryMap(grid):
    assert len(grid) <= 4
    zero = AffineConstantExpr.get(0, ctx)
    exprs = [AffineDimExpr.get(i, ctx) for i in range(len(grid))]
    while len(exprs) < 4:
        exprs.insert(0, zero)
    return AffineMap.get(len(grid), len(grid), exprs, ctx)


def createDeviceAttr(
    grid,
    physicalGrid=[8, 8],
    deviceStartIdx=0,
    virtToPhysicalMap=None,
    physicalToVirtMap=None,
    system_desc=None,
    mesh_shape=[1],
    mesh_topology=[],
):
    if system_desc is not None:
        return ttcore.ir.DeviceAttr.from_system_desc(ctx, system_desc, mesh_shape)
    virtToPhysicalMap = (
        virtToPhysicalMap
        if virtToPhysicalMap is not None
        else inferVirtToPhysicalMap(grid, physicalGrid, mesh=mesh_shape)
    )
    physicalToVirtMap = (
        physicalToVirtMap
        if physicalToVirtMap is not None
        else inferPhysicalToVirtMap(grid, physicalGrid, mesh=mesh_shape)
    )
    l1Map = inferMemoryMap(grid)
    dramMap = inferMemoryMap(grid)
    totalDevices = volume(mesh_shape)
    return ttcore.ir.DeviceAttr.get(
        ctx,
        grid,
        virtToPhysicalMap,
        physicalToVirtMap,
        l1Map,
        dramMap,
        mesh_shape,
        list(range(deviceStartIdx, deviceStartIdx + totalDevices)),
        mesh_topology,
    )


c = lambda v: AffineConstantExpr.get(v, ctx)
d = lambda v: AffineDimExpr.get(v, ctx)
amap = lambda ndims, results: AffineMap.get(ndims, 0, results, ctx)
floordiv = AffineFloorDivExpr.get

c0 = c(0)
d0 = d(0)
d1 = d(1)
d2 = d(2)

print("=== From SystemDesc ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1, chipIds = [0]>
print(
    "", createDeviceAttr([8, 8], system_desc=ttcore.ir.SystemDescAttr.get_default(ctx))
)

# ------------------------------------------------------------------------------

print("=== Simple single device ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1, chipIds = [0]>
print("", createDeviceAttr([8, 8]))

# ------------------------------------------------------------------------------

print("\n=== Data parallel over batch ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<2x8x8, virt_to_physical_map = (d0, d1, d2) -> (d0, d1, d2), physical_to_virt_map = (d0, d1, d2) -> (d0 mod 2, d1, d2)>, l1Map = [[M:.*]], dramMap = [[M:.*]], meshShape = 2x1x1, chipIds = [0, 1]>
print("divide batch by 2\n", createDeviceAttr([2, 8, 8], mesh_shape=[2, 1, 1]))
# CHECK: ttcore.device<workerGrid = #ttcore.grid<4x8x8, virt_to_physical_map = (d0, d1, d2) -> (d0, d1, d2), physical_to_virt_map = (d0, d1, d2) -> (d0 mod 4, d1, d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 4x1x1, chipIds = [0, 1, 2, 3]>
print("divide batch by 4\n", createDeviceAttr([4, 8, 8], mesh_shape=[4, 1, 1]))

# ------------------------------------------------------------------------------

print("\n=== Data parallel over 2d ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x16, virt_to_physical_map = (d0, d1) -> (d1 floordiv 8, d0, d1 mod 8), physical_to_virt_map = (d0, d1, d2) -> (d1, (d0 mod 2) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1x2, chipIds = [0, 1]>
print(
    "Reinterpret 2 devices as grid side by side, 1x2 mesh\n",
    createDeviceAttr([8, 16], mesh_shape=[1, 2]),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<16x8, virt_to_physical_map = (d0, d1) -> (d0 floordiv 8, d0 mod 8, d1), physical_to_virt_map = (d0, d1, d2) -> ((d0 mod 2) * 8 + d1, d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 2x1, chipIds = [0, 1]>
print(
    "Reinterpret 2 devices as grid top to bottom, 2x1 mesh\n",
    createDeviceAttr([16, 8], mesh_shape=[2, 1]),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<16x32, virt_to_physical_map = (d0, d1) -> (d1 floordiv 8 + (d0 floordiv 8) * 4, d0 mod 8, d1 mod 8), physical_to_virt_map = (d0, d1, d2) -> (((d0 floordiv 4) mod 2) * 8 + d1, (d0 mod 4) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 2x4, chipIds = [0, 1, 2, 3, 4, 5, 6, 7]>
print("8 devices 2x4 mesh\n", createDeviceAttr([16, 32], mesh_shape=[2, 4]))
# CHECK: ttcore.device<workerGrid = #ttcore.grid<32x16, virt_to_physical_map = (d0, d1) -> (d1 floordiv 8 + (d0 floordiv 8) * 2, d0 mod 8, d1 mod 8), physical_to_virt_map = (d0, d1, d2) -> (((d0 floordiv 2) mod 4) * 8 + d1, (d0 mod 2) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 4x2, chipIds = [0, 1, 2, 3, 4, 5, 6, 7]>
print("8 devices 4x2 mesh\n", createDeviceAttr([32, 16], mesh_shape=[4, 2]))

# ------------------------------------------------------------------------------

print("\n=== Data parallel over 2d and batch (3d) ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<2x8x16, virt_to_physical_map = (d0, d1, d2) -> (d0 * 2 + d2 floordiv 8, d1, d2 mod 8), physical_to_virt_map = (d0, d1, d2) -> ((d0 floordiv 2) mod 2, d1, (d0 mod 2) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 2x1x2, chipIds = [0, 1, 2, 3]>
print(
    "divide batch by 2, 2x1x2 mesh\n",
    createDeviceAttr([2, 8, 16], mesh_shape=[2, 1, 2]),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<3x24x8, virt_to_physical_map = (d0, d1, d2) -> (d0 * 3 + d1 floordiv 8, d1 mod 8, d2), physical_to_virt_map = (d0, d1, d2) -> ((d0 floordiv 3) mod 3, (d0 mod 3) * 8 + d1, d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 3x3x1, chipIds = [0, 1, 2, 3, 4, 5, 6, 7, 8]>
print(
    "divide batch by 3, 3x3x1 mesh\n",
    createDeviceAttr([3, 24, 8], mesh_shape=[3, 3, 1]),
)

# ------------------------------------------------------------------------------

print("\n=== nD ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<3x2x8x8, virt_to_physical_map = (d0, d1, d2, d3) -> (d0 * 2 + d1, d2, d3), physical_to_virt_map = (d0, d1, d2) -> ((d0 floordiv 2) mod 3, d0 mod 2, d1, d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 3x2x1x1, chipIds = [0, 1, 2, 3, 4, 5]>
print("", createDeviceAttr([3, 2, 8, 8], mesh_shape=[3, 2, 1, 1]))

# ------------------------------------------------------------------------------fix

print("\n=== Data parallel batch on single device ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<2x4x8, virt_to_physical_map = (d0, d1, d2) -> (0, d0 * 4 + d1, d2), physical_to_virt_map = (d0, d1, d2) -> (d1 floordiv 4, d1 mod 4, d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1, chipIds = [0]>
print(
    "divide batch by 2, top 4 rows get batch 0, bottom 4 rows get batch 1\n",
    createDeviceAttr(
        [2, 4, 8],
        virtToPhysicalMap=amap(3, [c0, d0 * 4 + d1, d2]),
        physicalToVirtMap=amap(3, [floordiv(d1, c(4)), d1 % 4, d2]),
    ),
)

# ------------------------------------------------------------------------------fix

print("\n=== Pipeline parallel ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<2x8x16, virt_to_physical_map = (d0, d1, d2) -> (d0 * 2 + d2 floordiv 8, d1, d2 mod 8), physical_to_virt_map = (d0, d1, d2) -> ((d0 floordiv 2) mod 2, d1, (d0 mod 2) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 2x1x2, chipIds = [0, 1, 2, 3]>
print(
    "view devices 0-3 in one way\n",
    createDeviceAttr([2, 8, 16], deviceStartIdx=0, mesh_shape=[2, 1, 2]),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<16x16, virt_to_physical_map = (d0, d1) -> (d1 floordiv 8 + (d0 floordiv 8) * 2, d0 mod 8, d1 mod 8), physical_to_virt_map = (d0, d1, d2) -> (((d0 floordiv 2) mod 2) * 8 + d1, (d0 mod 2) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 2x2, chipIds = [4, 5, 6, 7]>
print(
    "view devices 4-7 in another way\n",
    createDeviceAttr([16, 16], deviceStartIdx=4, mesh_shape=[2, 2]),
)

# ------------------------------------------------------------------------------

print("\n=== Reinterpreted Grids ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d1, d0), physical_to_virt_map = (d0, d1, d2) -> (d2, d1)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1, chipIds = [0]>
print(
    "transposed\n",
    createDeviceAttr(
        [8, 8],
        virtToPhysicalMap=amap(2, [c0, d1, d0]),
        physicalToVirtMap=amap(3, [d2, d1]),
    ),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<1x64, virt_to_physical_map = (d0, d1) -> (0, d0 * 8 + d1 floordiv 8, d1 mod 8), physical_to_virt_map = (d0, d1, d2) -> (d1 floordiv 8, (d1 mod 8) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1, chipIds = [0]>
print(
    "extra wide\n",
    createDeviceAttr(
        [1, 64],
        virtToPhysicalMap=amap(2, [c0, d0 * 8 + floordiv(d1, c(8)), d1 % 8]),
        physicalToVirtMap=amap(3, [floordiv(d1, c(8)), (d1 % 8) * 8 + d2]),
    ),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<64x1, virt_to_physical_map = (d0, d1) -> (0, d1 * 8 + d0 floordiv 8, d0 mod 8), physical_to_virt_map = (d0, d1, d2) -> ((d1 mod 8) * 8 + d2, d1 floordiv 8)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1, chipIds = [0]>
print(
    "extra tall transposed\n",
    createDeviceAttr(
        [64, 1],
        virtToPhysicalMap=amap(2, [c0, d1 * 8 + floordiv(d0, c(8)), d0 % 8]),
        physicalToVirtMap=amap(3, [(d1 % 8) * 8 + d2, floordiv(d1, c(8))]),
    ),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, (d0 + d1) mod 8), physical_to_virt_map = (d0, d1, d2) -> (d1, (d2 - d1) mod 8)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1, chipIds = [0]>
print(
    "staircase systolic\n",
    createDeviceAttr(
        [8, 8],
        virtToPhysicalMap=amap(2, [c0, d0, (d0 + d1) % 8]),
        physicalToVirtMap=amap(3, [d1, (d2 - d1) % 8]),
    ),
)

# ------------------------------------------------------------------------------

print("\n=== Mesh Topology ===")
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1, chipIds = [0], meshTopology = [ring]>
print(
    "single device ring topology\n",
    createDeviceAttr([8, 8], mesh_topology=[ttcore.ir.Topology.Ring]),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x16, virt_to_physical_map = (d0, d1) -> (d1 floordiv 8, d0, d1 mod 8), physical_to_virt_map = (d0, d1, d2) -> (d1, (d0 mod 2) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1x2, chipIds = [0, 1], meshTopology = [ring, linear]>
print(
    "1x2 mesh with ring and linear topology\n",
    createDeviceAttr(
        [8, 16],
        mesh_shape=[1, 2],
        mesh_topology=[ttcore.ir.Topology.Ring, ttcore.ir.Topology.Linear],
    ),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x16, virt_to_physical_map = (d0, d1) -> (d1 floordiv 8, d0, d1 mod 8), physical_to_virt_map = (d0, d1, d2) -> (d1, (d0 mod 2) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1x2, chipIds = [0, 1], meshTopology = [ring, ring]>
print(
    "1x2 mesh with ring topology on both axes\n",
    createDeviceAttr(
        [8, 16],
        mesh_shape=[1, 2],
        mesh_topology=[ttcore.ir.Topology.Ring, ttcore.ir.Topology.Ring],
    ),
)
# CHECK: ttcore.device<workerGrid = #ttcore.grid<8x16, virt_to_physical_map = (d0, d1) -> (d1 floordiv 8, d0, d1 mod 8), physical_to_virt_map = (d0, d1, d2) -> (d1, (d0 mod 2) * 8 + d2)>, l1Map = [[L1:.*]], dramMap = [[DRAM:.*]], meshShape = 1x2, chipIds = [0, 1], meshTopology = [ring, disabled]>
print(
    "1x2 mesh with ring and disabled topology\n",
    createDeviceAttr(
        [8, 16],
        mesh_shape=[1, 2],
        mesh_topology=[ttcore.ir.Topology.Ring, ttcore.ir.Topology.Disabled],
    ),
)
