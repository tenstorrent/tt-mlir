// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_constant attributes {} {
  func.func public @test_splat() -> tensor<64xf32> {
    %0 = stablehlo.constant dense<0.3> : tensor<64xf32>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e-01> : tensor<64xf32>}> : () -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }

  func.func public @test_multiple() -> tensor<2x2xf32> {
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  func.func public @test_scalar_int() -> tensor<i32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    return %0 : tensor<i32>
    // CHECK: return %{{[0-9]+}} : tensor<1xi32>
  }

  func.func public @test_scalar_float() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
    return %0 : tensor<f32>
    // CHECK: return %{{[0-9]+}} : tensor<1xf32>
  }
}
