// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // Regression for tt-mlir#9211: stablehlo.gather clamps OOB start indices to
  // [0, dim - slice_sizes] (spec); the ttir.gather path does not, so OOB
  // indices read garbage on device. The single-index direct-gather lowering
  // must clamp the normalized indices to [0, rows - 1].
  // CHECK-LABEL: func.func @gather_single_index_clamped
  func.func @gather_single_index_clamped(%operand: tensor<256xi64>, %start_indices: tensor<65536x1xi32>) -> tensor<65536xi64> {
    // CHECK: "ttir.clamp_scalar"
    // CHECK-SAME: max = 255 : i32
    // CHECK-SAME: min = 0 : i32
    // CHECK: "ttir.gather"
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<256xi64>, tensor<65536x1xi32>) -> tensor<65536xi64>
    return %0 : tensor<65536xi64>
  }

  // The 2D integer row-select shape clamps to the row count too.
  // CHECK-LABEL: func.func @gather_int_select_rows_clamped
  func.func @gather_int_select_rows_clamped(%operand: tensor<6x4xi32>, %start_indices: tensor<3x1xi32>) -> tensor<3x4xi32> {
    // CHECK: "ttir.clamp_scalar"
    // CHECK-SAME: max = 5 : i32
    // CHECK: "ttir.gather"
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 4>}> : (tensor<6x4xi32>, tensor<3x1xi32>) -> tensor<3x4xi32>
    return %0 : tensor<3x4xi32>
  }

  // Partial-slice (needsExpansion) gathers are unchanged: their spec clamp
  // bound is rows - slice_size per component, applied before expansion — not
  // covered by the slice-1 clamp.
  // CHECK-LABEL: func.func @gather_partial_slice_unclamped
  func.func @gather_partial_slice_unclamped(%operand: tensor<6x4xi32>, %start_indices: tensor<2x1xi32>) -> tensor<2x3x4xi32> {
    // CHECK-NOT: "ttir.clamp_scalar"
    // CHECK: "ttir.gather"
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 3, 4>}> : (tensor<6x4xi32>, tensor<2x1xi32>) -> tensor<2x3x4xi32>
    return %0 : tensor<2x3x4xi32>
  }

  // Multi-index gathers take the embedding path; no clamp there.
  // CHECK-LABEL: func.func @gather_multi_index_embedding_unclamped
  func.func @gather_multi_index_embedding_unclamped(%operand: tensor<1x12x12x768xbf16>, %start_indices: tensor<16x12x2xi32>) -> tensor<1x3x3x768x16x12xbf16> {
    // CHECK-NOT: "ttir.clamp_scalar"
    // CHECK: "ttir.embedding"
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 1, 2, 3], start_index_map = [1, 2], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 3, 3, 768>}> : (tensor<1x12x12x768xbf16>, tensor<16x12x2xi32>) -> tensor<1x3x3x768x16x12xbf16>
    return %0 : tensor<1x3x3x768x16x12xbf16>
  }
}
