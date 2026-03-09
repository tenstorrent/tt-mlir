// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Input A must be at least 4D
module attributes {} {
  func.func @sparse_matmul_input_a_rank(%a: tensor<32x2880xbf16>, %b: tensor<1x4x2880x5760xbf16>, %s: tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16> {
    %0 = "ttnn.sparse_matmul"(%a, %b, %s) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
    return %0 : tensor<2x4x1x4x32x5760xbf16>
  }
}
// CHECK: error: 'ttnn.sparse_matmul' op Input A must be at least a 4D tensor

// -----

// Input B must be 4D
module attributes {} {
  func.func @sparse_matmul_input_b_rank(%a: tensor<2x4x32x2880xbf16>, %b: tensor<2880x5760xbf16>, %s: tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16> {
    %0 = "ttnn.sparse_matmul"(%a, %b, %s) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16>, tensor<2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
    return %0 : tensor<2x4x1x4x32x5760xbf16>
  }
}
// CHECK: error: 'ttnn.sparse_matmul' op Input B must be a 4D tensor

// -----

// Sparsity must be 4D
module attributes {} {
  func.func @sparse_matmul_sparsity_rank(%a: tensor<2x4x32x2880xbf16>, %b: tensor<1x4x2880x5760xbf16>, %s: tensor<4xbf16>) -> tensor<2x4x1x4x32x5760xbf16> {
    %0 = "ttnn.sparse_matmul"(%a, %b, %s) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
    return %0 : tensor<2x4x1x4x32x5760xbf16>
  }
}
// CHECK: error: 'ttnn.sparse_matmul' op Sparsity tensor must be a 4D tensor

// -----

// At least one input must be sparse
module attributes {} {
  func.func @sparse_matmul_no_sparse(%a: tensor<2x4x32x2880xbf16>, %b: tensor<1x4x2880x5760xbf16>, %s: tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16> {
    %0 = "ttnn.sparse_matmul"(%a, %b, %s) <{is_input_a_sparse = false, is_input_b_sparse = false, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
    return %0 : tensor<2x4x1x4x32x5760xbf16>
  }
}
// CHECK: error: 'ttnn.sparse_matmul' op At least one of is_input_a_sparse or is_input_b_sparse must be true

// -----

// Input B first dimension must be 1
module attributes {} {
  func.func @sparse_matmul_b_dim0(%a: tensor<2x4x32x2880xbf16>, %b: tensor<2x4x2880x5760xbf16>, %s: tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16> {
    %0 = "ttnn.sparse_matmul"(%a, %b, %s) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16>, tensor<2x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
    return %0 : tensor<2x4x1x4x32x5760xbf16>
  }
}
// CHECK: error: 'ttnn.sparse_matmul' op Input B first dimension must be 1

// -----

// Output rank for dense-sparse must be 6D
module attributes {} {
  func.func @sparse_matmul_output_rank(%a: tensor<2x4x32x2880xbf16>, %b: tensor<1x4x2880x5760xbf16>, %s: tensor<2x4x1x4xbf16>) -> tensor<2x4x32x5760xbf16> {
    %0 = "ttnn.sparse_matmul"(%a, %b, %s) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x32x5760xbf16>
    return %0 : tensor<2x4x32x5760xbf16>
  }
}
// CHECK: error: 'ttnn.sparse_matmul' op Output must be 6D for dense-sparse mode
