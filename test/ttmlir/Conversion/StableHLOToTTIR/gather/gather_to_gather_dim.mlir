// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // Batched gather with offset dims: operand 256x24x3, indices 256x1x1 -> 256x1x3
  // This is a batched index-select along dim 1, keeping all of dim 2.
  // CHECK-LABEL: func.func @gather_batched_with_offset_dims
  func.func @gather_batched_with_offset_dims(%operand: tensor<256x24x3xf32>, %start_indices: tensor<256x1x1xi32>) -> tensor<256x1x3xf32> {
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.broadcast"
    // CHECK: "ttir.gather"
    // CHECK-SAME: {dim = 1 : i32}
    // CHECK-SAME: (tensor<256x24x3xf32>, tensor<256x1x3xi32>) -> tensor<256x1x3xf32>
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 3>}> : (tensor<256x24x3xf32>, tensor<256x1x1xi32>) -> tensor<256x1x3xf32>
    return %0 : tensor<256x1x3xf32>
  }

  // Batched gather without offset dims: operand 256x24, indices 256x1x1 -> 256x1
  // This is a batched index-select along dim 1 with no remaining offset dims.
  // CHECK-LABEL: func.func @gather_batched_no_offset_dims
  func.func @gather_batched_no_offset_dims(%operand: tensor<256x24xi1>, %start_indices: tensor<256x1x1xi32>) -> tensor<256x1xi1> {
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.gather"
    // CHECK-SAME: {dim = 1 : i32}
    // CHECK-SAME: (tensor<256x24xi1>, tensor<256x1xi32>) -> tensor<256x1xi1>
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<256x24xi1>, tensor<256x1x1xi32>) -> tensor<256x1xi1>
    return %0 : tensor<256x1xi1>
  }

  // Batched gather with multiple indices per batch.
  // CHECK-LABEL: func.func @gather_batched_multiple_indices
  func.func @gather_batched_multiple_indices(%operand: tensor<32x100x64xf32>, %start_indices: tensor<32x10x1xi32>) -> tensor<32x10x64xf32> {
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.broadcast"
    // CHECK: "ttir.gather"
    // CHECK-SAME: {dim = 1 : i32}
    // CHECK-SAME: (tensor<32x100x64xf32>, tensor<32x10x64xi32>) -> tensor<32x10x64xf32>
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 64>}> : (tensor<32x100x64xf32>, tensor<32x10x1xi32>) -> tensor<32x10x64xf32>
    return %0 : tensor<32x10x64xf32>
  }

  // index_vector_dim=1 on 2D indices, offset_dims=[1] instead of collapsed_slice_dims.
  // CHECK-LABEL: func.func @gather_batched_2d_indices_with_offset
  func.func @gather_batched_2d_indices_with_offset(%operand: tensor<256x24xi1>, %start_indices: tensor<256x1xi32>) -> tensor<256x1xi1> {
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.gather"
    // CHECK-SAME: {dim = 1 : i32}
    // CHECK-SAME: (tensor<256x24xi1>, tensor<256x1xi32>) -> tensor<256x1xi1>
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], operand_batching_dims = [0], start_indices_batching_dims = [0], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<256x24xi1>, tensor<256x1xi32>) -> tensor<256x1xi1>
    return %0 : tensor<256x1xi1>
  }
}
