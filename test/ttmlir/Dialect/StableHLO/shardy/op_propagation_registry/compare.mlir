// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @compare_with_shard_on_operands(%arg0: tensor<4x56x56x96xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<4x56x56x96xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> tensor<4x56x56x96xi1> {
  %0 = stablehlo.compare  EQ, %arg0, %arg1 : (tensor<4x56x56x96xbf16>, tensor<4x56x56x96xbf16>) -> tensor<4x56x56x96xi1>
  return %0 : tensor<4x56x56x96xi1>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: %1 = stablehlo.compare  EQ, %arg2, %arg3 : (tensor<2x56x56x96xbf16>, tensor<2x56x56x96xbf16>) -> tensor<2x56x56x96xi1>
// CHECK: sdy.return %1 : tensor<2x56x56x96xi1>

func.func @compare_with_different_comparison_directions(%arg0: tensor<4x56x56x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<4x56x56x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> (tensor<4x56x56x96xi1>, tensor<4x56x56x96xi1>, tensor<4x56x56x96xi1>) {
  %0 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<4x56x56x96xf32>, tensor<4x56x56x96xf32>) -> tensor<4x56x56x96xi1>
  %1 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<4x56x56x96xf32>, tensor<4x56x56x96xf32>) -> tensor<4x56x56x96xi1>
  %2 = stablehlo.compare  NE, %arg0, %arg1 : (tensor<4x56x56x96xf32>, tensor<4x56x56x96xf32>) -> tensor<4x56x56x96xi1>
  return %0, %1, %2 : tensor<4x56x56x96xi1>, tensor<4x56x56x96xi1>, tensor<4x56x56x96xi1>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: %1 = stablehlo.compare  LT, %arg2, %arg3 : (tensor<2x56x56x96xf32>, tensor<2x56x56x96xf32>) -> tensor<2x56x56x96xi1>
// CHECK: %2 = stablehlo.compare  GT, %arg2, %arg3 : (tensor<2x56x56x96xf32>, tensor<2x56x56x96xf32>) -> tensor<2x56x56x96xi1>
// CHECK: %3 = stablehlo.compare  NE, %arg2, %arg3 : (tensor<2x56x56x96xf32>, tensor<2x56x56x96xf32>) -> tensor<2x56x56x96xi1>
// CHECK: sdy.return %1, %2, %3 : tensor<2x56x56x96xi1>, tensor<2x56x56x96xi1>, tensor<2x56x56x96xi1>
