// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_logit attributes {} {
  func.func public @test_logit(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.logistic %arg0 : tensor<13x21x3xf32>
    // CHECK: [[VAL0:%[0-9]+]] = ttir.empty() : [[TENSOR_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.sigmoid"(%arg0, [[VAL0]]) : ([[TENSOR_SIZE]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xf32>
  }
}
