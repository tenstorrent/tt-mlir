// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_rsqrt attributes {} {
  func.func public @test_cbrt(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.cbrt %arg0 : tensor<4xf64>
    // CHECK: = ttir.empty
    // CHECK: = "ttir.cbrt"
    return %0 : tensor<4xf64>
  }
}
