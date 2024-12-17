// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline  %s | FileCheck %s
module @jit_eltwise_convert attributes {} {
  func.func public @test_convert(%arg0: tensor<2x4xf32>) -> tensor<2x4xbf16> {
    %0 = stablehlo.convert %arg0 : (tensor<2x4xf32>) -> tensor<2x4xbf16>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.typecast"
    // CHECK-SAME: (tensor<2x4xf32>, tensor<2x4xbf16>) -> tensor<2x4xbf16>
    return %0 : tensor<2x4xbf16>
  }
}

module @jit_eltwise_add attributes {} {
  func.func public @test_add(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.convert %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[ARG1:.*]] = "ttir.typecast"[[C:.*]]
    %1 = stablehlo.convert %arg1 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[ARG2:.*]] = "ttir.typecast"[[C:.*]]
    %2 = stablehlo.add %0, %1 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = "ttir.add"(%[[ARG1]], %[[ARG2]],
    return %2 : tensor<13x21x3xf32>
  }
}
