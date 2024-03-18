// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.     ■ Too many errors emitted, stopping now
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTPASSES_TD
#define TTMLIR_TTMLIR_TTPASSES_TD

include "mlir/Pass/PassBase.td"

def TTPackGeneric: Pass<"tt-pack-generic", "::mlir::ModuleOp"> {
  let summary = "Tensor pack all generic ops.";
  let description = [{
    ```
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%0 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<64x128xf32>

    // Gets transformed to:

    %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<64x128xf32> -> tensor<2x4x32x32xf32>
    %2 = tensor.empty() : tensor<2x4x32x32xf32>
    %pack_0 = tensor.pack %arg1 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<64x128xf32> -> tensor<2x4x32x32xf32>
    %3 = tensor.empty() : tensor<2x4x32x32xf32>
    %pack_1 = tensor.pack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %3 : tensor<64x128xf32> -> tensor<2x4x32x32xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%pack, %pack_0 : tensor<2x4x32x32xf32>, tensor<2x4x32x32xf32>) outs(%pack_1 
: tensor<2x4x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      linalg.yield %5 : f32
    } -> tensor<2x4x32x32xf32>
    %unpack = tensor.unpack %4 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<2x4x32x32xf32> -> tensor<64x128xf32>
    ```
  }];
}

def TTTilize: Pass<"tt-tilize", "::mlir::ModuleOp"> {
  let summary = "Tilizes tensors.";
  let description = [{
    Tilizes tensors.
  }];
}

#endif
