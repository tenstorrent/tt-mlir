// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_reduce_maximum attributes {} {
  func.func public @test_reduce_maximum(%arg0: tensor<128x10xf32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [0, 1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<f32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.max"
    // CHECK-SAME: tensor<128x10xf32>,
    // CHECK-SAME: tensor<1xf32>
    // CEHCK-SAME: tensor<1xf32>
    return %0 : tensor<f32>
  }
}
