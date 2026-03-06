// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_gather attributes {} {
  func.func public @test_gather_0(%operand: tensor<32000x1024xf32>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xf32> {
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1024>}> : (tensor<32000x1024xf32>, tensor<1x32xi32>) -> tensor<1x32x1024xf32>
    // CHECK: = "ttir.gather"
    return %0 : tensor<1x32x1024xf32>
  }

  func.func public @test_gather_1(%operand: tensor<448x384xf32>, %start_indices: tensor<1x2x1xi32>) -> tensor<1x2x384xf32> {
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 384>}> : (tensor<448x384xf32>, tensor<1x2x1xi32>) -> tensor<1x2x384xf32>
    // CHECK: = "ttir.gather"
    return %0 : tensor<1x2x384xf32>
  }

  func.func public @test_gather_2(%operand: tensor<51864x384xf32>, %start_indices: tensor<1x2xi32>) -> tensor<1x2x384xf32> {
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 384>}> : (tensor<51864x384xf32>, tensor<1x2xi32>) -> tensor<1x2x384xf32>
    // CHECK: = "ttir.gather"
    return %0 : tensor<1x2x384xf32>
  }

  func.func public @test_gather_3(%arg0: tensor<32128x512xbf16>, %arg1: tensor<1x15xi64>) -> tensor<1x15x512xbf16> {
    // CHECK: %[[VAL:[0-9]+]] = "ttir.gather"(%arg0, %arg1)
    // CHECK-SAME: collapsed_slice_dims = array<i64: 0>,
    // CHECK-SAME: index_vector_dim = 2 : si64,
    // CHECK-SAME: indices_are_sorted = false,
    // CHECK-SAME: offset_dims = array<i64: 2>,
    // CHECK-SAME: operand_batching_dims = array<i64>,
    // CHECK-SAME: slice_sizes = array<i64: 1, 512>,
    // CHECK-SAME: start_index_map = array<i64: 0>,
    // CHECK-SAME: start_indices_batching_dims = array<i64>
    // CHECK-SAME: (tensor<32128x512xbf16>, tensor<1x15xi64>) -> tensor<1x15x512xbf16>
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 512>}> : (tensor<32128x512xbf16>, tensor<1x15xi64>) -> tensor<1x15x512xbf16>
    // CHECK: return %[[VAL]] : tensor<1x15x512xbf16>
    return %0 : tensor<1x15x512xbf16>
  }
}
