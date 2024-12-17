// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_reduce_add attributes {} {
  func.func public @test_reduce_add(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64xf32>{
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.add across dimensions = [2, 3] : (tensor<1x64x112x112xf32>, tensor<f32>) -> tensor<1x64xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.sum"[[C:.*]]
    return %0 : tensor<1x64xf32>
  }
}
