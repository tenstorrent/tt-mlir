// REQUIRES: stablehlo
// RUN: ttmlir-opt --automatic-sharding-pipeline="mesh-shape=2,4" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=2, "batch"=4]>

func.func public @sharding_mismatch(%arg0: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}, %arg1: tensor<784x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}, %arg2: tensor<8192x2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x784xf32>, tensor<784x2048xf32>) -> tensor<8192x2048xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<8192x2048xf32>
  return %1 : tensor<8192x2048xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {}]>, <@mesh, [{"batch"}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: stablehlo.all_gather
