// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s

// A scatter whose update window spans the very dimension being scattered
// along: nothing is collapsed, so the index says where the window starts along
// dimension 1 and the window then runs on from there.
//
//   operand[i, idx + j] = updates[i, j]
//
// The index tensor is therefore the broadcast start plus an iota of the
// within-window offsets.

module @scatter_window_dim {
  // CHECK-LABEL: func.func @column_run
  // CHECK: %[[IOTA:.*]] = "ttir.arange"()
  // CHECK-SAME: arange_dimension = 1
  // CHECK: ttir.add
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 1
  func.func @column_run(%operand: tensor<6400x16xf64>, %indices: tensor<1xi64>, %updates: tensor<6400x3xf64>) -> tensor<6400x16xf64> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      "stablehlo.return"(%arg1) : (tensor<f64>) -> ()
    }) : (tensor<6400x16xf64>, tensor<1xi64>, tensor<6400x3xf64>) -> tensor<6400x16xf64>
    return %0 : tensor<6400x16xf64>
  }

  // operand[0, idx[j]] = updates[j, 0]
  //
  // The scattered dimension is a window dimension and carries the batch of
  // window starts at the same time, which is legal because the window is a
  // single element wide. It contributes no offset of its own, so there must be
  // no iota here - adding one would write to idx[j] + j.
  // CHECK-LABEL: func.func @window_carries_batch
  // CHECK-NOT: ttir.arange
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 1
  func.func @window_carries_batch(%operand: tensor<2x3xi32>, %indices: tensor<4x1xi32>, %updates: tensor<4x1xi32>) -> tensor<2x3xi32> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "stablehlo.return"(%arg1) : (tensor<i32>) -> ()
    }) : (tensor<2x3xi32>, tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  // A single-column window is the same shape of lowering, just with a
  // degenerate iota.
  // CHECK-LABEL: func.func @single_column
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 1
  func.func @single_column(%operand: tensor<6400x16xf64>, %indices: tensor<1xi64>, %updates: tensor<6400x1xf64>) -> tensor<6400x16xf64> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      "stablehlo.return"(%arg1) : (tensor<f64>) -> ()
    }) : (tensor<6400x16xf64>, tensor<1xi64>, tensor<6400x1xf64>) -> tensor<6400x16xf64>
    return %0 : tensor<6400x16xf64>
  }
}
