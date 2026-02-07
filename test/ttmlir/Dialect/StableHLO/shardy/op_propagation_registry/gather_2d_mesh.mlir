// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=2,4" %s 2>&1 | FileCheck %s

module {
  sdy.mesh @mesh = <["_axis_0"=2, "_axis_1"=4]>
  func.func @gather_allreduce_overlap_bug(
      %arg0: tensor<32x64xi1> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>},
      %arg1: tensor<32x64x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>}
  ) -> tensor<32x64xi1> {
    // CHECK: stablehlo.all_gather
    // CHECK-SAME: (tensor<16x64
    // CHECK-SAME: -> tensor<32x64x
    // CHECK: stablehlo.gather
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<32x64xi1>, tensor<32x64x2xi64>) -> tensor<32x64xi1>
    return %0 : tensor<32x64xi1>
  }
}
