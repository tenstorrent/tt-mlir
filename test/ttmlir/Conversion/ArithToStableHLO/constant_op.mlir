// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_constant attributes {} {
  func.func public @test_scalar_float() -> tensor<f32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = arith.constant dense<3.0> : tensor<f32>
    // CHECK: return %{{[0-9]+}} : tensor<1xf32>
    return %0 : tensor<f32>
  }

  func.func public @test_splat_float() -> tensor<64xf32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %0 = arith.constant dense<3.0> : tensor<64xf32>
    // CHECK: return %{{[0-9]+}} : tensor<64xf32>
    return %0 : tensor<64xf32>
  }

  func.func public @test_multiple_float() -> tensor<2x2xf32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    %0 = arith.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  func.func public @test_scalar_int() -> tensor<i32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = arith.constant dense<3> : tensor<i32>
    // CHECK: return %{{[0-9]+}} : tensor<1xi32>
    return %0 : tensor<i32>
  }

  func.func public @test_splat_int() -> tensor<64xi32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xi32>}> : () -> tensor<64xi32>
    %0 = arith.constant dense<3> : tensor<64xi32>
    // CHECK: return %{{[0-9]+}} : tensor<64xi32>
    return %0 : tensor<64xi32>
  }

  func.func public @test_multiple_int() -> tensor<2x2xi32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  func.func public @test_scalar_uint() -> tensor<ui32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xui32>}> : () -> tensor<1xui32>
    %0 = arith.constant dense<3> : tensor<ui32>
    // CHECK: return %{{[0-9]+}} : tensor<1xui32>
    return %0 : tensor<ui32>
  }

  func.func public @test_splat_uint() -> tensor<64xui32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xui32>}> : () -> tensor<64xui32>
    %0 = arith.constant dense<3> : tensor<64xui32>
    // CHECK: return %{{[0-9]+}} : tensor<64xui32>
    return %0 : tensor<64xui32>
  }

  func.func public @test_multiple_uint() -> tensor<2x2xui32> {
     // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xui32>}> : () -> tensor<2x2xui32>
    %0 = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xui32>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xui32>
    return %0 : tensor<2x2xui32>
  }
}
