// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @gather_simple(%arg0: tensor<4x56x56x96xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<56xi64>) -> tensor<4x56x56x96xbf16> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1, 56, 96>}> : (tensor<4x56x56x96xbf16>, tensor<56xi64>) -> tensor<4x56x56x96xbf16>
  return %0 : tensor<4x56x56x96xbf16>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: "stablehlo.gather"(%arg2, %arg3) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1, 56, 96>}> : (tensor<2x56x56x96xbf16>, tensor<56xi64>) -> tensor<2x56x56x96xbf16>
// CHECK: sdy.return %1 : tensor<2x56x56x96xbf16>
