// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s

// Single-dimensional scatters whose index tensor has no batch dimensions, so
// index_vector_dim is 0 rather than 1 and there is exactly one window to write.
// inserted_window_dims also leaves the update tensor a rank short of the
// operand, so it has to be reshaped up to the operand's rank.

module @scatter_index_vector_dim_zero {
  // operand[:, idx] = updates
  // CHECK-LABEL: func.func @column_2d
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 1
  func.func @column_2d(%operand: tensor<2x3xi32>, %indices: tensor<1xi32>, %updates: tensor<2xi32>) -> tensor<2x3xi32> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "stablehlo.return"(%arg1) : (tensor<i32>) -> ()
    }) : (tensor<2x3xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  // operand[:, :, idx] = updates
  // CHECK-LABEL: func.func @plane_3d
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 2
  func.func @plane_3d(%operand: tensor<2x3x3xf64>, %indices: tensor<1xi32>, %updates: tensor<2x3xf64>) -> tensor<2x3x3xf64> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      "stablehlo.return"(%arg1) : (tensor<f64>) -> ()
    }) : (tensor<2x3x3xf64>, tensor<1xi32>, tensor<2x3xf64>) -> tensor<2x3x3xf64>
    return %0 : tensor<2x3x3xf64>
  }

  // operand[idx] = scalar update
  // CHECK-LABEL: func.func @scalar_1d
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 0
  func.func @scalar_1d(%operand: tensor<2xi64>, %indices: tensor<1xi64>, %updates: tensor<i64>) -> tensor<2xi64> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      "stablehlo.return"(%arg1) : (tensor<i64>) -> ()
    }) : (tensor<2xi64>, tensor<1xi64>, tensor<i64>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }
}
