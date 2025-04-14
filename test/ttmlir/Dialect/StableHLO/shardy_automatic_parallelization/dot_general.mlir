// RUN: ttmlir-opt --shardy-automatic-parallelization="mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @dot_general_no_batching_dims(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] : (tensor<4x32xf32>, tensor<32x16xf32>) -> tensor<4x16xf32>
// CHECK: sdy.return %1 : tensor<4x16xf32>

func.func @dot_general_batching_dims(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}, %arg1: tensor<4x32x16xf32>) -> tensor<4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}]>, <@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: stablehlo.dot_general %arg2, %arg3, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x8x32xf32>, tensor<2x32x16xf32>) -> tensor<2x8x16xf32>
// CHECK: sdy.return %1 : tensor<2x8x16xf32>
