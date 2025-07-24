// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// CHECK-LABEL: module {
module {
  // CHECK: sdy.mesh @mesh = <["x"=1, "y"=1]>
  func.func @single_chip(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
      %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
      %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
      return %2 : tensor<1x128xf32>
  }
}

// -----

// CHECK-LABEL: module {
module {
  // CHECK: sdy.mesh @mesh = <["x"=1, "y"=2]>
  func.func @main(%arg0: tensor<256x256xf32> {mhlo.sharding = "{devices=[1,2]<=[2]}"}) -> (tensor<256x256xf32> {jax.result_info = "result", mhlo.sharding = "{devices=[1,2]<=[2]}"}) {
    %0 = stablehlo.negate %arg0 : tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}
