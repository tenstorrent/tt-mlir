// REQUIRES: stablehlo
// RUN: ttmlir-opt --automatic-sharding-pipeline="mesh-shape=1,2 automatic-arg-analysis" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func public @abs(%arg0: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.abs %arg0 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: stablehlo.abs %arg1 : tensor<16x48x24x32xf32>
// CHECK: sdy.return %1 : tensor<16x48x24x32xf32>

func.func public @add(%arg0: tensor<32x48x24x32xf32>, %arg1: tensor<32x48x24x32xf32>) -> tensor<32x48x24x32xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<32x48x24x32xf32>
  return %0 : tensor<32x48x24x32xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: stablehlo.add %arg2, %arg3 : tensor<16x48x24x32xf32>
// CHECK: sdy.return %1 : tensor<16x48x24x32xf32>
