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
// ttir.scatter needs an absolute position per element, so the index tensor is
// the broadcast start plus an iota of the within-window offsets. Shapes are
// from the ORB model, where the Pade approximant writes a run of columns into
// a wider buffer.

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
