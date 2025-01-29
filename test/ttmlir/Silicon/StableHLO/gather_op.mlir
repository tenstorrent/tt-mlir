// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RU1N: FileCheck --input-file=%t.mlir %s

module @jit_gather attributes {} {
  func.func public @test_gather_0(%operand: tensor<32000x1024xbf16>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xbf16> {
    // CHECK-LABEL: func.func public @test_gather_0
    // CHECK: ttnn.empty
    // CHECK: ttnn.embedding
    // CHECK-SAME: tensor<1x32xi32,
    // CHECK-SAME: tensor<1x32x1024xbf16
    // CHECK-SAME: tensor<32000x1024xbf16,
    // CHECK-SAME: -> tensor<1x32x1024xbf16
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1024>}> : (tensor<32000x1024xbf16>, tensor<1x32xi32>) -> tensor<1x32x1024xbf16>
    return %0 : tensor<1x32x1024xbf16>
  }

  func.func public @test_gather_1(%operand: tensor<51864x384xbf16>, %start_indices: tensor<1x2xi32>) -> tensor<1x2x384xbf16> {
    // CHECK-LABEL: func.func public @test_gather_1
    // CHECK: ttnn.empty
    // CHECK: ttnn.embedding
    // CHECK-SAME: tensor<1x2xi32,
    // CHECK-SAME: tensor<1x2x384xbf16
    // CHECK-SAME: tensor<51864x384xbf16,
    // CHECK-SAME: -> tensor<1x2x384xbf16
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 384>}> : (tensor<51864x384xbf16>, tensor<1x2xi32>) -> tensor<1x2x384xbf16>
    return %0 : tensor<1x2x384xbf16>
  }

  func.func public @test_gather_2(%operand: tensor<32128x512xbf16>, %start_indices: tensor<1x15xi64>) -> tensor<1x15x512xbf16> {
    // CHECK-LABEL: func.func public @test_gather_2
    // CHECK: ttnn.empty
    // CHECK: ttnn.embedding
    // CHECK-SAME: tensor<1x16xi32,
    // CHECK-SAME: tensor<1x15x512xbf16
    // CHECK-SAME: tensor<32128x512xbf16,
    // CHECK-SAME: -> tensor<1x15x512xbf16
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 512>}> : (tensor<32128x512xbf16>, tensor<1x15xi64>) -> tensor<1x15x512xbf16>
    return %0 : tensor<1x15x512xbf16>
  }
}
