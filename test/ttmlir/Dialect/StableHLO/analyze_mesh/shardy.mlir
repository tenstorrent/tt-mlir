// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // CHECK: sdy.mesh @mesh = <["batch_updated"=1, "batch"=8]>
  sdy.mesh @mesh = <["batch"=8]>
  func.func public @main(%arg0: tensor<1024x2x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}) -> (tensor<2048x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1024x2x32x32xf32>) -> tensor<2048x1024xf32>
    return %0 : tensor<2048x1024xf32>
  }
}

// -----
module {
  // CHECK: sdy.mesh @empty_mesh = <["default_updated"=1, "default"=1]>
  sdy.mesh @empty_mesh = <[]>
  func.func public @main(%arg0: tensor<784x128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}, %arg1: tensor<128xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}, %arg2: tensor<128x784xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}, {}]>}) -> tensor<128x128xf32> {
    %0 = stablehlo.dot_general %arg2, %arg0, contracting_dims = [1] x [0] : (tensor<128x784xf32>, tensor<784x128xf32>) -> tensor<128x128xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<128x128xf32>
    %3 = stablehlo.add %0, %2 : tensor<128x128xf32>
    return %3 : tensor<128x128xf32>
  }
}
