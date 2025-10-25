# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

# Tests for the d2m::CBType Python bindings

from ttmlir.ir import *
from ttmlir.dialects import d2m, ttcore

with Context() as ctx, Location.unknown():
    # Test CBType.get() with a tensor type
    f32 = F32Type.get()
    tensor_type = RankedTensorType.get([32, 64], f32)
    cb_type = d2m.ir.CBType.get(ctx, tensor_type)

    # CHECK: !d2m.cb<tensor<32x64xf32>>
    print(cb_type)

    # Test CBType.cast() and get_underlying()
    casted = d2m.ir.CBType.cast(cb_type)
    underlying = casted.get_underlying()

    # CHECK: tensor<32x64xf32>
    print(underlying)

    # Test CBType with memref
    memref_type = MemRefType.get([16, 32], f32)
    cb_memref = d2m.ir.CBType.get(ctx, memref_type)

    # CHECK: !d2m.cb<memref<16x32xf32>>
    print(cb_memref)

    # TileType test
    tile_type = ttcore.ir.TileType.get(ctx, 32, 32, 2)  # 2 = BFloat16
    tile_tensor = RankedTensorType.get([2, 4], tile_type)
    cb_tile = d2m.ir.CBType.get(ctx, tile_tensor)

    # CHECK: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, bf16>>>
    print(cb_tile)
