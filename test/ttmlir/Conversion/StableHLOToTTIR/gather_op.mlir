// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_gather attributes {} {
  func.func public @test_gather_0(%operand: tensor<32000x1024xf32>, %start_indices: tensor<1x32xi32>) -> tensor<1x32x1024xf32> {
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1024>}> : (tensor<32000x1024xf32>, tensor<1x32xi32>) -> tensor<1x32x1024xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.embedding"[[C:.*]]
    return %0 : tensor<1x32x1024xf32>
  }
  func.func public @test_gather_1(%operand: tensor<448x384xf32>, %start_indices: tensor<1x2x1xi64>) -> tensor<1x2x384xf32> {
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 384>}> : (tensor<448x384xf32>, tensor<1x2x1xi64>) -> tensor<1x2x384xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.embedding"[[C:.*]]
    return %0 : tensor<1x2x384xf32>
  }

  func.func public @test_gather_2(%operand: tensor<51864x384xf32>, %start_indices: tensor<1x2xi64>) -> tensor<1x2x384xf32> {
    %0 = "stablehlo.gather"(%operand, %start_indices) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 384>}> : (tensor<51864x384xf32>, tensor<1x2xi64>) -> tensor<1x2x384xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.embedding"[[C:.*]]
    return %0 : tensor<1x2x384xf32>
  }

}
