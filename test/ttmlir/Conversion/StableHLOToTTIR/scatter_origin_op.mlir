// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s

// Scatters with an empty `scatter_dims_to_operand_dims`. Every start index is
// then 0, so the single update window is written at the operand's origin; when
// it also covers the whole operand the op is a plain overwrite and folds away
// to the update tensor. JAX emits this for an update whose index array turned
// out to be statically empty, hence the `tensor<0x...>` index operands.

module @scatter_origin {
  // CHECK-LABEL: func.func @overwrite_1d
  // CHECK-NOT: ttir.scatter
  // CHECK: return %arg2
  func.func @overwrite_1d(%operand: tensor<256xi64>, %indices: tensor<0xi64>, %updates: tensor<256xi64>) -> tensor<256xi64> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      "stablehlo.return"(%arg1) : (tensor<i64>) -> ()
    }) : (tensor<256xi64>, tensor<0xi64>, tensor<256xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }

  // CHECK-LABEL: func.func @overwrite_3d
  // CHECK-NOT: ttir.scatter
  // CHECK: return %arg2
  func.func @overwrite_3d(%operand: tensor<2x3x3xf64>, %indices: tensor<0xi32>, %updates: tensor<2x3x3xf64>) -> tensor<2x3x3xf64> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      "stablehlo.return"(%arg1) : (tensor<f64>) -> ()
    }) : (tensor<2x3x3xf64>, tensor<0xi32>, tensor<2x3x3xf64>) -> tensor<2x3x3xf64>
    return %0 : tensor<2x3x3xf64>
  }
}
