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

  func.func public @test_f64_scalar() -> tensor<f64> {
    // CHECK: %[[VAL:[0-9]+]] = "ttir.constant"() <{value = dense<3.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = stablehlo.constant dense<0.3> : tensor<f64>
    // CHECK: return %[[VAL]] : tensor<1xf32>
    return %0 : tensor<f64>
  }

  func.func public @test_f64_splat() -> tensor<64xf64> {
    // CHECK: %[[VAL:[0-9]+]] = "ttir.constant"() <{value = dense<3.000000e-01> : tensor<64xf32>}> : () -> tensor<64xf32>
    %0 = stablehlo.constant dense<0.3> : tensor<64xf64>
    // CHECK: return %[[VAL]] : tensor<64xf32>
    return %0 : tensor<64xf64>
  }

  func.func public @test_f64_multiple() -> tensor<2x2xf64> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %[[VAL:[0-9]+]] = "ttir.constant"() <{value = dense<{{([[])}}[0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    %0 = stablehlo.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf64>
    // CHECK: return %[[VAL]] : tensor<2x2xf32>
    return %0 : tensor<2x2xf64>
  }

  func.func public @test_f64_inf() -> tensor<f64> {
    // CHECK: %[[VAL:[0-9]+]] = "ttir.constant"() <{value = dense<0xFF800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    // CHECK: return %[[VAL]] : tensor<1xf32>
    return %0 : tensor<f64>
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

  func.func public @test_uint8_scalar() -> tensor<ui8> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xui8>}> : () -> tensor<1xui8>
    %0 = stablehlo.constant dense<3> : tensor<ui8>
    // CHECK: return %{{[0-9]+}} : tensor<1xui8>
    return %0 : tensor<ui8>
  }

  func.func public @test_uint8_splat() -> tensor<64xui8> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xui8>}> : () -> tensor<64xui8>
    %0 = stablehlo.constant dense<3> : tensor<64xui8>
    // CHECK: return %{{[0-9]+}} : tensor<64xui8>
    return %0 : tensor<64xui8>
  }

  func.func public @test_uint8_multiple() -> tensor<2x2xui8> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xui8>}> : () -> tensor<2x2xui8>
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xui8>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xui8>
    return %0 : tensor<2x2xui8>
  }

  func.func public @test_uint16_scalar() -> tensor<ui16> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xui16>}> : () -> tensor<1xui16>
    %0 = stablehlo.constant dense<3> : tensor<ui16>
    // CHECK: return %{{[0-9]+}} : tensor<1xui16>
    return %0 : tensor<ui16>
  }

  func.func public @test_uint16_splat() -> tensor<64xui16> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xui16>}> : () -> tensor<64xui16>
    %0 = stablehlo.constant dense<3> : tensor<64xui16>
    // CHECK: return %{{[0-9]+}} : tensor<64xui16>
    return %0 : tensor<64xui16>
  }

  func.func public @test_uint16_multiple() -> tensor<2x2xui16> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xui16>}> : () -> tensor<2x2xui16>
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xui16>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xui16>
    return %0 : tensor<2x2xui16>
  }

  func.func public @test_uint32_scalar() -> tensor<ui32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xui32>}> : () -> tensor<1xui32>
    %0 = stablehlo.constant dense<3> : tensor<ui32>
    // CHECK: return %{{[0-9]+}} : tensor<1xui32>
    return %0 : tensor<ui32>
  }

  func.func public @test_uint32_splat() -> tensor<64xui32> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xui32>}> : () -> tensor<64xui32>
    %0 = stablehlo.constant dense<3> : tensor<64xui32>
    // CHECK: return %{{[0-9]+}} : tensor<64xui32>
    return %0 : tensor<64xui32>
  }

  func.func public @test_uint32_multiple() -> tensor<2x2xui32> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xui32>}> : () -> tensor<2x2xui32>
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xui32>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xui32>
    return %0 : tensor<2x2xui32>
  }

  func.func public @test_uint64_scalar() -> tensor<ui64> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<1xui32>}> : () -> tensor<1xui32>
    %0 = stablehlo.constant dense<3> : tensor<ui64>
    // CHECK: return %{{[0-9]+}} : tensor<1xui32>
    return %0 : tensor<ui64>
  }

  func.func public @test_uint64_splat() -> tensor<64xui64> {
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<3> : tensor<64xui32>}> : () -> tensor<64xui32>
    %0 = stablehlo.constant dense<3> : tensor<64xui64>
    // CHECK: return %{{[0-9]+}} : tensor<64xui32>
    return %0 : tensor<64xui64>
  }

  func.func public @test_uint64_multiple() -> tensor<2x2xui64> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %{{[0-9]+}} = "ttir.constant"() <{value = dense<{{([[])}}[0, 1], [2, 3]]> : tensor<2x2xui32>}> : () -> tensor<2x2xui32>
    %0 = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xui64>
    // CHECK: return %{{[0-9]+}} : tensor<2x2xui32>
    return %0 : tensor<2x2xui64>
  }

  func.func public @test_int8_negative_scalar() -> tensor<i8> {
    // CHECK-LABEL: func.func public @test_int8_negative_scalar
    // CHECK: %[[CONSTANT:[0-9]+]] = "ttir.constant"() <{value = dense<-3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = stablehlo.constant dense<-3> : tensor<i8>
    // CHECK: return %[[CONSTANT]] : tensor<1xi8>
    return %0 : tensor<i8>
  }

  func.func public @test_int16_negative_splat() -> tensor<64xi16> {
    // CHECK-LABEL: func.func public @test_int16_negative_splat
    // CHECK: %[[CONSTANT:[0-9]+]] = "ttir.constant"() <{value = dense<-3> : tensor<64xi16>}> : () -> tensor<64xi16>
    %0 = stablehlo.constant dense<-3> : tensor<64xi16>
    // CHECK: return %[[CONSTANT]] : tensor<64xi16>
    return %0 : tensor<64xi16>
  }

  func.func public @test_int32_negative_multiple() -> tensor<2x2xi32> {
    // The ugly regex after `dense` is necessary because double square opening
    // brackets indicate substitution block in FileCheck syntax.
    // CHECK: %[[CONSTANT:[0-9]+]] = "ttir.constant"() <{value = dense<{{([[])}}[0, -1], [-2, 3]]> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
    %0 = stablehlo.constant dense<[[0, -1], [-2, 3]]> : tensor<2x2xi32>
    // CHECK: return %[[CONSTANT]] : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }

  func.func public @test_int64_negative_min_scalar() -> tensor<i64> {
    // CHECK-LABEL: func.func public @test_int64_negative_min_scalar
    // CHECK: %[[CONSTANT:[0-9]+]] = "ttir.constant"() <{value = dense<-2147483648> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = stablehlo.constant dense<9223372036854775808> : tensor<i64>
    // CHECK: return %[[CONSTANT]] : tensor<1xi32>
    return %0 : tensor<i64>
  }
}
