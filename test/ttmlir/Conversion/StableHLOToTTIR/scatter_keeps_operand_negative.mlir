// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: not ttmlir-opt --convert-stablehlo-to-ttir %s 2>&1 | FileCheck %s

// An update computation that returns its *first* argument, the operand, rather
// than the update. The scatter therefore leaves the operand untouched.
//
// getReduceTypeFromRegion reports ReduceType::Invalid for this, exactly as it
// does for a real overwrite - it only looks for an arithmetic op in the region
// and finds none either way. Lowering on the reduce type alone would turn this
// into an overwrite and silently produce the wrong values, so the returned
// value has to be checked as well.

// CHECK: failed to legalize operation 'stablehlo.scatter'
module @scatter_keeps_operand {
  func.func @main(%operand: tensor<2x3xi32>, %indices: tensor<1xi32>, %updates: tensor<2xi32>) -> tensor<2x3xi32> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      "stablehlo.return"(%arg0) : (tensor<i32>) -> ()
    }) : (tensor<2x3xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<2x3xi32>
    return %0 : tensor<2x3xi32>
  }
}
