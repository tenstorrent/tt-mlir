// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s

// Scatters with batching dimensions, which StableHLO uses for the vmap of a
// scatter: operand dimension 0 is batched against index dimension 0, so batch
// element i only ever writes into operand[i]. A batching dimension needs no
// index arithmetic - it only has to be laid out at the operand dimension it
// batches over.

module @scatter_batching_dims {
  // operand[i, idx[i, j]] = updates[i, j]
  // CHECK-LABEL: func.func @batched_2d
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 1
  func.func @batched_2d(%operand: tensor<2x3xi32>, %indices: tensor<2x2x1xi32>, %updates: tensor<2x2xi32>) -> tensor<2x3xi32> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "stablehlo.return"(%arg1) : (tensor<i32>) -> ()
    }) : (tensor<2x3xi32>, tensor<2x2x1xi32>, tensor<2x2xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }

  // operand[i, idx[i, j], k] = updates[i, j, k]
  // CHECK-LABEL: func.func @batched_3d
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 1
  func.func @batched_3d(%operand: tensor<2x3x3xf64>, %indices: tensor<2x2x1xi32>, %updates: tensor<2x2x3xf64>) -> tensor<2x3x3xf64> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      "stablehlo.return"(%arg1) : (tensor<f64>) -> ()
    }) : (tensor<2x3x3xf64>, tensor<2x2x1xi32>, tensor<2x2x3xf64>) -> tensor<2x3x3xf64>
    return %0 : tensor<2x3x3xf64>
  }

  // operand[i, 0] = updates[i]
  //
  // Batching dimensions with an empty `scatter_dims_to_operand_dims`: nothing is
  // read out of the index vector, which is why it has extent 0. The scattered
  // dimension then defaults to 0 - here the batching dimension - and the iota
  // supplies the batch coordinate it needs.
  // CHECK-LABEL: func.func @batched_origin
  // CHECK: %[[IOTA:.*]] = "ttir.arange"()
  // CHECK-SAME: arange_dimension = 0
  // CHECK: ttir.scatter
  // CHECK-SAME: dim = 0
  func.func @batched_origin(%operand: tensor<2x3xi32>, %indices: tensor<2x0xi32>, %updates: tensor<2xi32>) -> tensor<2x3xi32> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "stablehlo.return"(%arg1) : (tensor<i32>) -> ()
    }) : (tensor<2x3xi32>, tensor<2x0xi32>, tensor<2xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}
