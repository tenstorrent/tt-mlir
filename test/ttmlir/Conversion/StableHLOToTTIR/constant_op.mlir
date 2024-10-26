// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_constant attributes {} {
  func.func public @test_splat_float() -> tensor<64xf32> {
    %0 = stablehlo.constant dense<0.3> : tensor<64xf32>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e-01> : tensor<64xf32>}> : () -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }

  func.func public @test_multiple_float() -> tensor<2x2xf32> {
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  func.func public @test_scalar_float() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
    return %0 : tensor<f32>
    // CHECK: return %{{[0-9]+}} : tensor<1xf32>
  }

  func.func public @test_splat_int() -> tensor<64xi32> {
    %0 = stablehlo.constant dense<3> : tensor<64xi32>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xi32>}> : () -> tensor<64xi32>
    return %0 : tensor<64xi32>
  }

  func.func public @test_multiple_int() -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  func.func public @test_scalar_int() -> tensor<i32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    return %0 : tensor<i32>
    // CHECK: return %{{[0-9]+}} : tensor<1xi32>
  }

  func.func public @test_splat_boolean() -> tensor<2x2xi1> {
    %0 = stablehlo.constant dense<true> : tensor<2x2xi1>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<1> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    return %0 : tensor<2x2xi1>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xi32>
  }

  func.func public @test_multi_boolean() -> tensor<2x2xi1> {
    %0 = stablehlo.constant dense<[[true, false], [false, true]]> : tensor<2x2xi1>
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[1, 0], [0, 1]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    return %0 : tensor<2x2xi1>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xi32>
  }

  func.func public @test_scalar_boolean() -> tensor<i1> {
    %0 = stablehlo.constant dense<true> : tensor<i1>
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<255> : tensor<1xi32>}> : () -> tensor<1xi32>
    return %0 : tensor<i1>
    // CHECK: return %{{[0-9]+}} : tensor<1xi32>
  }
}
