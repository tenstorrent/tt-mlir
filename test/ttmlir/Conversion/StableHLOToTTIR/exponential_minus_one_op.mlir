// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_expm1 attributes {} {
  func.func public @test_expm1(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.exponential_minus_one %arg0 : tensor<13x21x3xf32>
    // CHECK: [[VAL0:%[0-9]+]] = ttir.empty() : [[TENSOR_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.expm1"(%arg0, [[VAL0]]) : ([[TENSOR_SIZE]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xf32>
    // CHECK: return [[VAL1]] : [[TENSOR_SIZE]]
  }
}
