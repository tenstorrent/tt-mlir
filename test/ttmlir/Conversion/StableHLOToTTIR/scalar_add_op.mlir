// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_scalar_add attributes {} {
  func.func public @test_scalar_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    // CHECK: = ttir.empty
    // CHECK: = "ttir.add"
    return %0 : tensor<f32>
  }
}
