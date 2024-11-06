// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_constant attributes {} {
  func.func public @test_boolean_scalar() -> tensor<i1> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %0 = stablehlo.constant dense<true> : tensor<i1>
    // CHECK: return %{{[0-9]+}} : tensor<1xbf16>
    return %0 : tensor<i1>
  }

  func.func public @test_boolean_splat() -> tensor<64xi1> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<64xbf16>}> : () -> tensor<64xbf16>
    %0 = stablehlo.constant dense<true> : tensor<64xi1>
    // CHECK: return %{{[0-9]+}} : tensor<64xbf16>
    return %0 : tensor<64xi1>
  }

  func.func public @test_boolean_multiple() -> tensor<2x2xi1> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[1.000000e+00, 0.000000e+00], [0.000000e+00, 1.000000e+00]]> : tensor<2x2xbf16>}> : () -> tensor<2x2xbf16>
    %0 = stablehlo.constant dense<[[true, false], [false, true]]> : tensor<2x2xi1>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xbf16>
    return %0 : tensor<2x2xi1>
  }

  func.func public @test_bfloat16_scalar() -> tensor<bf16> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %0 = stablehlo.constant dense<3.0> : tensor<bf16>
    // CHECK: return %{{[0-9]+}} : tensor<1xbf16>
    return %0 : tensor<bf16>
  }

  func.func public @test_bfloat16_splat() -> tensor<64xbf16> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<64xbf16>}> : () -> tensor<64xbf16>
    %0 = stablehlo.constant dense<3.0> : tensor<64xbf16>
    // CHECK: return %{{[0-9]+}} : tensor<64xbf16>
    return %0 : tensor<64xbf16>
  }

  func.func public @test_bfloat16_multiple() -> tensor<2x2xbf16> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xbf16>}> : () -> tensor<2x2xbf16>
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xbf16>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xbf16>
    return %0 : tensor<2x2xbf16>
  }

  func.func public @test_float16_scalar() -> tensor<f16> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %0 = stablehlo.constant dense<3.0> : tensor<f16>
    // CHECK: return %{{[0-9]+}} : tensor<1xf16>
    return %0 : tensor<f16>
  }

  func.func public @test_float16_splat() -> tensor<64xf16> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<64xf16>}> : () -> tensor<64xf16>
    %0 = stablehlo.constant dense<3.0> : tensor<64xf16>
    // CHECK: return %{{[0-9]+}} : tensor<64xf16>
    return %0 : tensor<64xf16>
  }

  func.func public @test_float16_multiple() -> tensor<2x2xf16> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf16>}> : () -> tensor<2x2xf16>
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf16>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xf16>
    return %0 : tensor<2x2xf16>
  }

  func.func public @test_float_scalar() -> tensor<f32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    // CHECK: return %{{[0-9]+}} : tensor<1xf32>
    return %0 : tensor<f32>
  }

  func.func public @test_float_splat() -> tensor<64xf32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3.000000e-01> : tensor<64xf32>}> : () -> tensor<64xf32>
    %0 = stablehlo.constant dense<0.3> : tensor<64xf32>
    // CHECK: return %{{[0-9]+}} : tensor<64xf32>
    return %0 : tensor<64xf32>
  }

  func.func public @test_float_multiple() -> tensor<2x2xf32> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  func.func public @test_int8_scalar() -> tensor<i8> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = stablehlo.constant dense<3> : tensor<i8>
    // CHECK: return %{{[0-9]+}} : tensor<1xi8>
    return %0 : tensor<i8>
  }

  func.func public @test_int8_splat() -> tensor<64xi8> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xi8>}> : () -> tensor<64xi8>
    %0 = stablehlo.constant dense<3> : tensor<64xi8>
    // CHECK: return %{{[0-9]+}} : tensor<64xi8>
    return %0 : tensor<64xi8>
  }

  func.func public @test_int8_multiple() -> tensor<2x2xi8> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xi8>}> : () -> tensor<2x2xi8>
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi8>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xi8>
    return %0 : tensor<2x2xi8>
  }

  func.func public @test_int16_scalar() -> tensor<i16> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xi16>}> : () -> tensor<1xi16>
    %0 = stablehlo.constant dense<3> : tensor<i16>
    // CHECK: return %{{[0-9]+}} : tensor<1xi16>
    return %0 : tensor<i16>
  }

  func.func public @test_int16_splat() -> tensor<64xi16> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xi16>}> : () -> tensor<64xi16>
    %0 = stablehlo.constant dense<3> : tensor<64xi16>
    // CHECK: return %{{[0-9]+}} : tensor<64xi16>
    return %0 : tensor<64xi16>
  }

  func.func public @test_int16_multiple() -> tensor<2x2xi16> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xi16>}> : () -> tensor<2x2xi16>
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi16>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xi16>
    return %0 : tensor<2x2xi16>
  }

  func.func public @test_int32_scalar() -> tensor<i32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = stablehlo.constant dense<3> : tensor<i32>
    // CHECK: return %{{[0-9]+}} : tensor<1xi32>
    return %0 : tensor<i32>
  }

  func.func public @test_int32_splat() -> tensor<64xi32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xi32>}> : () -> tensor<64xi32>
    %0 = stablehlo.constant dense<3> : tensor<64xi32>
    // CHECK: return %{{[0-9]+}} : tensor<64xi32>
    return %0 : tensor<64xi32>
  }

  func.func public @test_int32_multiple() -> tensor<2x2xi32> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  func.func public @test_int64_scalar() -> tensor<i64> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = stablehlo.constant dense<3> : tensor<i64>
    // CHECK: return %{{[0-9]+}} : tensor<1xi32>
    return %0 : tensor<i64>
  }

  func.func public @test_int64_splat() -> tensor<64xi64> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xi32>}> : () -> tensor<64xi32>
    %0 = stablehlo.constant dense<3> : tensor<64xi64>
    // CHECK: return %{{[0-9]+}} : tensor<64xi32>
    return %0 : tensor<64xi64>
  }

  func.func public @test_int64_multiple() -> tensor<2x2xi64> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xi32>
    return %0 : tensor<2x2xi64>
  }
}
