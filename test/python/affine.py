# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

from ttmlir.ir import *

ctx = Context()


def updiv(n, d):
    return (n + d - 1) // d


def volume(shape):
    vol = 1
    for dim in shape:
        vol *= dim
    return vol


def inferAffineMap(grid, physicalGrid=[8, 8]):
    assert len(grid) >= 2
    mesh = grid[:-2] + [
        updiv(grid[-2], physicalGrid[-2]),
        updiv(grid[-1], physicalGrid[-1]),
    ]
    totalDevices = getTotalDevices(grid, physicalGrid=physicalGrid)
    p0 = AffineConstantExpr.get(physicalGrid[0], ctx)
    p1 = AffineConstantExpr.get(physicalGrid[1], ctx)
    dZ = AffineConstantExpr.get(0, ctx)
    dY = AffineDimExpr.get(len(grid) - 2, ctx)
    dX = AffineDimExpr.get(len(grid) - 1, ctx)
    if totalDevices > 1:
        v = mesh[-1]
        dZ = AffineFloorDivExpr.get(dY, p0) * v + AffineFloorDivExpr.get(dX, p1)
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


c = lambda v: AffineConstantExpr.get(v, ctx)
d = lambda v: AffineDimExpr.get(v, ctx)
amap = lambda ndims, results: AffineMap.get(ndims, 0, list(results), ctx)
floordiv = AffineFloorDivExpr.get

c0 = c(0)
c1 = c(1)
d0 = d(0)
d1 = d(1)
d2 = d(2)
d3 = d(3)
d4 = d(4)
d5 = d(5)
d6 = d(6)
d7 = d(7)

print(amap(3, [d0, d1, d2]))
