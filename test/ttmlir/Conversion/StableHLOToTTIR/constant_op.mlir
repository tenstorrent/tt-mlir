// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_constant attributes {} {
  func.func public @test_splat() -> tensor<64xf32> {
    %0 = stablehlo.constant dense<0.3> : tensor<64xf32>
    // CHECK: %[[C:.*]] = "ttir.constant"[[C:.*]]
    return %0 : tensor<64xf32>
  }

  func.func public @test_multiple() -> tensor<2x2xf32> {
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    // CHECK: %[[C:.*]] = "ttir.constant"[[C:.*]]
    return %0 : tensor<2x2xf32>
  }
}
