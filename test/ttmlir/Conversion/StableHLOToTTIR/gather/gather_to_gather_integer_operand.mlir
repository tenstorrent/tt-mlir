// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // 2D integer operand, single indexed dim with full trailing slice (D != 1),
  // so indices are broadcast across the weight dim before the gather.
  // CHECK-LABEL: func.func @gather_int_select_rows
  func.func @gather_int_select_rows(%operand: tensor<6x4xi32>, %start_indices: tensor<3x1xi32>) -> tensor<3x4xi32> {
    // CHECK-NOT: "ttir.embedding"
    // CHECK: "ttir.gather"
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<6x4xi32>, tensor<3x1xi32>) -> tensor<3x4xi32>
    return %0 : tensor<3x4xi32>
  }

  // 1D i64 operand (the Qwen 2.5 VL cu_seqlens shape): gathering scalar
  // elements (D == 1). Wide i64 values must not round through bf16.
  // CHECK-LABEL: func.func @gather_int_scalar_lookup
  func.func @gather_int_scalar_lookup(%operand: tensor<4xi64>, %start_indices: tensor<2x1xi64>) -> tensor<2xi64> {
    // CHECK-NOT: "ttir.embedding"
    // CHECK: "ttir.gather"
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<4xi64>, tensor<2x1xi64>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }

  // Float operand with the same single-index shape still uses ttir.embedding
  // (the fast vocab-lookup path is unchanged).
  // CHECK-LABEL: func.func @gather_float_stays_embedding
  func.func @gather_float_stays_embedding(%operand: tensor<6x4xbf16>, %start_indices: tensor<3x1xi32>) -> tensor<3x4xbf16> {
    // CHECK: "ttir.embedding"
    // CHECK-NOT: "ttir.gather"
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<6x4xbf16>, tensor<3x1xi32>) -> tensor<3x4xbf16>
    return %0 : tensor<3x4xbf16>
  }

  // 2D integer operand, single indexed dim with a partial (non-full, non-1)
  // slice, so start indices are expanded to gather the implied consecutive
  // rows (needsExpansion == true). Must stay on ttir.gather to keep integer
  // precision rather than rounding through the embedding bf16 weight.
  // CHECK-LABEL: func.func @gather_int_partial_slice
  func.func @gather_int_partial_slice(%operand: tensor<6x4xi32>, %start_indices: tensor<2x1xi32>) -> tensor<2x3x4xi32> {
    // CHECK-NOT: "ttir.embedding"
    // CHECK: "ttir.gather"
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 3, 4>}> : (tensor<6x4xi32>, tensor<2x1xi32>) -> tensor<2x3x4xi32>
    return %0 : tensor<2x3x4xi32>
  }
}
