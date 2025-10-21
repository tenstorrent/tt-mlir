// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @jit_neg_shardy1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=2, "y"=4]>
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}, {}]>}) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"x"}, {}, {}]>] out_shardings=[<@mesh, [{}, {"x"}, {}, {}]>] manual_axes={"x", "y"} (%arg1: tensor<1x512x128x1024xf32>) {
      %1 = stablehlo.negate %arg1 : tensor<1x512x128x1024xf32>
      sdy.return %1 : tensor<1x512x128x1024xf32>
    } : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    return %0 : tensor<1x1024x128x1024xf32>
  }
}

// CHECK: sdy.mesh @mesh = <["x"=2, "y"=4]>
// CHECK: %0 = sdy.manual_computation
