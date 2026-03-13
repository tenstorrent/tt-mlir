// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Step 2: Gather result element type is set to data input element type.
// Gather with input/result same type is unchanged. Gather with result type
// differing from input (e.g. after upstream) would be rewritten; ttir.gather
// verifier requires same element type so we only test unchanged case here.
// =============================================================================

// -----
// CHECK-LABEL: func.func @step2_gather_result_unchanged
// CHECK: "ttir.gather"(%arg0, %arg1) {{.*}} : (tensor<5x4xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
func.func @step2_gather_result_unchanged(%arg0: tensor<5x4xi32>, %arg1: tensor<2x1xi32>) -> tensor<2x4xi32> {
  %0 = "ttir.gather"(%arg0, %arg1) <{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 1 : si64, indices_are_sorted = false, offset_dims = array<i64: 1>, operand_batching_dims = array<i64>, slice_sizes = array<i64: 1, 4>, start_index_map = array<i64: 0>, start_indices_batching_dims = array<i64>}> : (tensor<5x4xi32>, tensor<2x1xi32>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}
