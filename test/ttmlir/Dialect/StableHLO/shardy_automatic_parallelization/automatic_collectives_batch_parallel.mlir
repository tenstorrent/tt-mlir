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
