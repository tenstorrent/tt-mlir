// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,2 argument-types=op_sequence=input,parameter,input" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func public @op_sequence(%arg0: tensor<32x1x8192x256xf32>, %arg1: tensor<256x2048xf32>, %arg2: tensor<32x1x8192x2048xf32>) -> (tensor<32x1x8192x2048xf32> {jax.result_info = ""}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x1x8192x256xf32>, tensor<256x2048xf32>) -> tensor<32x1x8192x2048xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<32x1x8192x2048xf32>
  %2 = stablehlo.multiply %1, %arg2 : tensor<32x1x8192x2048xf32>
  return %2 : tensor<32x1x8192x2048xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}, {}]>, <@mesh, [{"batch"}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: %1 = stablehlo.dot_general %arg3, %arg4, contracting_dims = [3] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x1x8192x256xf32>, tensor<256x2048xf32>) -> tensor<16x1x8192x2048xf32>
// CHECK: %2 = stablehlo.add %1, %arg5 : tensor<16x1x8192x2048xf32>
// CHECK: %3 = stablehlo.multiply %2, %arg5 : tensor<16x1x8192x2048xf32>
// CHECK: sdy.return %3 : tensor<16x1x8192x2048xf32>
