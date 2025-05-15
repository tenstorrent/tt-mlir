// REQUIRES: stablehlo
// RUN: ttmlir-opt --automatic-sharding-pipeline="mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func public @sharding_constraint(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"batch"}]>}) -> (tensor<32x32xf32> {jax.result_info = "", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}]> : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"model"}, {"batch"}]>] out_shardings=[<@mesh, [{}, {}]>]
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.all_gather
// CHECK: sdy.return %2 : tensor<32x32xf32>

func.func public @sharding_mismatch(%arg0: tensor<1024x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg1: tensor<256x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg2: tensor<256x256xf32>) -> (tensor<1024x256xf32> {jax.result_info = "result"}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x256xf32>) -> tensor<1024x256xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x256xf32>) -> tensor<1024x256xf32>
  return %1 : tensor<1024x256xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.dot_general %1, %2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512x256xf32>, tensor<256x256xf32>) -> tensor<512x256xf32>
// CHECK: stablehlo.dot_general %3, %arg5, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<512x256xf32>, tensor<256x256xf32>) -> tensor<512x256xf32>
// CHECK: sdy.return %4 : tensor<512x256xf32>
