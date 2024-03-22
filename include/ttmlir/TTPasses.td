// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.     ■ Too many errors emitted, stopping now
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTPASSES_TD
#define TTMLIR_TTMLIR_TTPASSES_TD

include "mlir/Pass/PassBase.td"

def TTTilize: Pass<"tt-tilize", "::mlir::ModuleOp"> {
  let summary = "Tensor tilize all generic ops.";
  let description = [{
    ```
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%0 : tensor<64x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<64x128xf32>

    // Gets transformed to:
    todo
    ```
  }];
}

def TTParallelize: Pass<"tt-parallelize", "::mlir::ModuleOp"> {
  let summary = "Parallelize all tensors and ops.";
  let description = [{
    todo
  }];
}

#endif
