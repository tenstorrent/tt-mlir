// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s

// An origin scatter (empty scatter_dims_to_operand_dims) whose window covers
// only part of the operand. Elements 4..255 keep their original values, so this
// is not an overwrite and must not fold to the update tensor.
//
// It still lowers: the window starts at the origin, so the position of each
// element along the scattered dimension is just its own offset within the
// window, which an iota supplies. Every other dimension is already addressed
// positionally by ttir.scatter.

// CHECK-LABEL: func.func @main
// The update tensor must not simply be returned.
// CHECK-NOT: return %arg2
// CHECK: ttir.arange
// CHECK: ttir.scatter
// CHECK-SAME: dim = 0
module @scatter_origin_partial_window {
  func.func @main(%operand: tensor<256xi64>, %indices: tensor<0xi64>, %updates: tensor<4xi64>) -> tensor<256xi64> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      "stablehlo.return"(%arg1) : (tensor<i64>) -> ()
    }) : (tensor<256xi64>, tensor<0xi64>, tensor<4xi64>) -> tensor<256xi64>
    return %0 : tensor<256xi64>
  }
}
