// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func public @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
    return %2 : tensor<1x128xf32>
}

// CHECK: sdy.mesh @mesh = <["x"=1, "y"=1]>
