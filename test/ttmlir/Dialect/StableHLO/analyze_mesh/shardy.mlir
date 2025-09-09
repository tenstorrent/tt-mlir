// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @jit_reshape attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1, "batch"=8]>
  func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> (tensor<2048x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
    return %0 : tensor<2048x1024xf32>
  }
}

// CHECK: sdy.mesh @mesh = <["x"=1, "batch"=8]>
