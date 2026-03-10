// REQUIRES: stablehlo
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// -----
// CHECK-LABEL: module {
module {
  sdy.mesh @mesh = <["x"=1, "batch"=8]>
  func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> (tensor<2048x1024xf32>) {
    %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
    return %0 : tensor<2048x1024xf32>
  }
}

// CHECK: ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128x2x32x32xf32>>, ttcore.shard_status = #ttcore.shard_status<presharded>
// CHECK: ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x1024xf32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>
