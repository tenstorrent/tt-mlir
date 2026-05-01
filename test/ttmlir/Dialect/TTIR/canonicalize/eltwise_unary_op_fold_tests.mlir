// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @abs_float_positive() -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @abs_float_positive
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 4.000000e+00
    // CHECK-NOT: "ttir.abs"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 4.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %1 = "ttir.abs"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @abs_float_negative() -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @abs_float_negative
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 4.000000e+00
    // CHECK-NOT: "ttir.abs"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -4.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %1 = "ttir.abs"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @abs_float_zero() -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @abs_float_zero
    // CHECK: "ttir.zeros"
    // CHECK-NOT: "ttir.abs"
    %0 = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    %1 = "ttir.abs"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @abs_int_positive() -> tensor<64x64xsi32> {
    // CHECK-LABEL: func.func @abs_int_positive
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 2
    // CHECK-NOT: "ttir.abs"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 2 : i32}> : () -> tensor<64x64xsi32>
    %1 = "ttir.abs"(%0) : (tensor<64x64xsi32>) -> tensor<64x64xsi32>
    return %1 : tensor<64x64xsi32>
  }

  func.func @abs_int_negative() -> tensor<64x64xsi32> {
    // CHECK-LABEL: func.func @abs_int_negative
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 2
    // CHECK-NOT: "ttir.abs"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -2 : i32}> : () -> tensor<64x64xsi32>
    %1 = "ttir.abs"(%0) : (tensor<64x64xsi32>) -> tensor<64x64xsi32>
    return %1 : tensor<64x64xsi32>
  }

  func.func @abs_int_zero() -> tensor<64x64xsi32> {
    // CHECK-LABEL: func.func @abs_int_zero
    // CHECK: "ttir.zeros"
    // CHECK-NOT: "ttir.abs"
    %0 = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xsi32>
    %1 = "ttir.abs"(%0) : (tensor<64x64xsi32>) -> tensor<64x64xsi32>
    return %1 : tensor<64x64xsi32>
  }

  func.func @abs_bf() -> tensor<64x64xbf16> {
    // CHECK-LABEL: func.func @abs_bf
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 4.000000e+00
    // CHECK-NOT: "ttir.abs"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -4.000000e+00 : f32}> : () -> tensor<64x64xbf16>
    %1 = "ttir.abs"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @abs_int_to_float_no_fold() -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @abs_int_to_float_no_fold
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -2
    // CHECK: "ttir.abs"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -2 : i32}> : () -> tensor<64x64xsi32>
    %1 = "ttir.abs"(%0) : (tensor<64x64xsi32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @abs_float_to_int_no_fold() -> tensor<64x64xsi32> {
    // CHECK-LABEL: func.func @abs_float_to_int_no_fold
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -2.000000e+00
    // CHECK: "ttir.abs"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -2.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %1 = "ttir.abs"(%0) : (tensor<64x64xf32>) -> tensor<64x64xsi32>
    return %1 : tensor<64x64xsi32>
  }

  func.func @atan0() -> tensor<32xf32> {
    // CHECK-LABEL: func.func @atan0
    // CHECK: "ttir.zeros"
    // CHECK-NOT: "ttir.atan"
    %0 = "ttir.zeros"() <{shape = array<i32: 32>}> : () -> tensor<32xf32>
    %1 = "ttir.atan"(%0) : (tensor<32xf32>) -> tensor<32xf32>
    return %1 : tensor<32xf32>
  }

  func.func @atan1() -> tensor<32xf32> {
    // CHECK-LABEL: func.func @atan1
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 0.7853981{{[0-9]*(e+00)?}} : f32
    // CHECK-NOT: "ttir.atan"
    %0 = "ttir.ones"() <{shape = array<i32: 32>}> : () -> tensor<32xf32>
    %1 = "ttir.atan"(%0) : (tensor<32xf32>) -> tensor<32xf32>
    return %1 : tensor<32xf32>
  }

  func.func @atan_bf() -> tensor<64x64xbf16> {
    // CHECK-LABEL: func.func @atan_bf
    // CHECK: "ttir.zeros"
    // CHECK-NOT: "ttir.atan"
    %0 = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xbf16>
    %1 = "ttir.atan"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @bitwise_not_int() -> tensor<64x64xsi32> {
    // CHECK-LABEL: func.func @bitwise_not_int
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 31
    // CHECK-NOT: "ttir.bitwise_not"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 0xFFFFFFE0 : i32}> : () -> tensor<64x64xsi32>
    %1 = "ttir.bitwise_not"(%0) : (tensor<64x64xsi32>) -> tensor<64x64xsi32>
    return %1 : tensor<64x64xsi32>
  }

  func.func @cbrt_float() -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @cbrt_float
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 2.000000e+00
    // CHECK-NOT: "ttir.cbrt"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 8.000000e+00 : f32}> : () -> tensor<64x64xf32>
    %1 = "ttir.cbrt"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @ceil_float() -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @ceil_float
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 2.000000e+00
    // CHECK-NOT: "ttir.ceil"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.200000e+00 : f32}> : () -> tensor<64x64xf32>
    %1 = "ttir.ceil"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @ceil_bf_same() -> tensor<64x64xbf16> {
    // CHECK-LABEL: func.func @ceil_bf_same
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 2.000000e+00
    // CHECK-NOT: "ttir.ceil"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 2.000000e+00 : f32}> : () -> tensor<64x64xbf16>
    %1 = "ttir.ceil"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @floor_bf() -> tensor<64x64xbf16> {
    // CHECK-LABEL: func.func @floor_bf
    // CHECK: "ttir.ones"
    // CHECK-NOT: "ttir.floor"
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 1.800000e+00 : f32}> : () -> tensor<64x64xbf16>
    %1 = "ttir.floor"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @floor_float_same() -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @floor_float_same
    // CHECK: "ttir.ones"
    // CHECK-NOT: "ttir.floor"
    %0 = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    %1 = "ttir.floor"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @clamp_scalar_uint() -> tensor<1x8xui8> {
    // CHECK-LABEL: func.func @clamp_scalar_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}2, 2, 2, 3, 4, 5, 5, 5]]>
    // CHECK-NOT: "ttir.clamp_scalar"
    %0 = "ttir.constant"() <{value = dense<[[0, 1, 2, 3, 4, 5, 6, 255]]> : tensor<1x8xui8>}> : () -> tensor<1x8xui8>
    %1 = "ttir.clamp_scalar"(%0) <{min = 2.000000e+00 : f32, max = 5.000000e+00 : f32}> : (tensor<1x8xui8>) -> tensor<1x8xui8>
    return %1 : tensor<1x8xui8>
  }

  func.func @clamp_scalar_sint() -> tensor<1x8xsi32> {
    // CHECK-LABEL: func.func @clamp_scalar_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}2, 2, 2, 3, 4, 5, 5, 2]]>
    // CHECK-NOT: "ttir.clamp_scalar"
    %0 = "ttir.constant"() <{value = dense<[[0, 1, 2, 3, 4, 5, 6, -1]]> : tensor<1x8xsi32>}> : () -> tensor<1x8xsi32>
    %1 = "ttir.clamp_scalar"(%0) <{min = 2 : i32, max = 5 : i32}> : (tensor<1x8xsi32>) -> tensor<1x8xsi32>
    return %1 : tensor<1x8xsi32>
  }

  func.func @clamp_scalar_bf() -> tensor<1x7xbf16> {
    // CHECK-LABEL: func.func @clamp_scalar_bf
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}2.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 5.000000e+00]]>
    // CHECK-NOT: "ttir.clamp_scalar"
    %0 = "ttir.constant"() <{value = dense<[[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<1x7xbf16>}> : () -> tensor<1x7xbf16>
    %1 = "ttir.clamp_scalar"(%0) <{min = 2 : i32, max = 5 : i32}> : (tensor<1x7xbf16>) -> tensor<1x7xbf16>
    return %1 : tensor<1x7xbf16>
  }

  func.func @clamp_scalar_f32() -> tensor<1x7xf32> {
    // CHECK-LABEL: func.func @clamp_scalar_f32
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}2.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 5.000000e+00]]>
    // CHECK-NOT: "ttir.clamp_scalar"
    %0 = "ttir.constant"() <{value = dense<[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]> : tensor<1x7xf32>}> : () -> tensor<1x7xf32>
    %1 = "ttir.clamp_scalar"(%0) <{min = 2 : i32, max = 5 : i32}> : (tensor<1x7xf32>) -> tensor<1x7xf32>
    return %1 : tensor<1x7xf32>
  }

  func.func @cos() -> tensor<2xf32> {
    // CHECK-LABEL: func.func @cos
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}1.0000{{[0-9]*(e\+00)?}}, -0.12884{{[0-9]*(e\+00)?}}]>
    // CHECK-NOT: "ttir.cos"
    %0 = "ttir.constant"() <{value = dense<[0.0, 1.7]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %1 = "ttir.cos"(%0) : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  func.func @exp() -> tensor<2xf32> {
    // CHECK-LABEL: func.func @exp
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}2.7182{{[0-9]*(e\+00)?}}, 12.182{{[0-9]*(e\+00)?}}]>
    // CHECK-NOT: "ttir.exp"
    %0 = "ttir.constant"() <{value = dense<[1.0, 2.5]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %1 = "ttir.exp"(%0) : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  func.func @expm1() -> tensor<5xf32> {
    // CHECK-LABEL: func.func @expm1
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[}}0.0000{{[0-9]*(e\+00)?}}, 1.7182{{[0-9]*(e\+00)?}}, 1.0000{{[0-9]*[eE]}}-10, 1.0000{{[0-9]*[eE]-0?}}7, 1.0000{{[0-9]*[eE]-0?}}5]>
    // CHECK-NOT: "ttir.expm1"
    %0 = "ttir.constant"() <{value = dense<[0.0, 1.0, 1.0e-10, 1.0e-7, 1.0e-5]> : tensor<5xf32>}> : () -> tensor<5xf32>
    %1 = "ttir.expm1"(%0) : (tensor<5xf32>) -> tensor<5xf32>
    return %1 : tensor<5xf32>
  }

  func.func @isfinite() -> tensor<5xf32> {
    // CHECK-LABEL: func.func @isfinite
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]>
    // CHECK-NOT: "ttir.isfinite"
    // %0 = [1, inf, -inf, qNaN, -2.33]
    %0 = "ttir.constant"() <{value = dense<[1.0, 0x7F800000, 0xFF800000, 0xFFC00001, -2.33]> : tensor<5xf32>}> : () -> tensor<5xf32>
    %1 = "ttir.isfinite"(%0) : (tensor<5xf32>) -> tensor<5xf32>
    return %1 : tensor<5xf32>
  }

  func.func @log1p() -> tensor<3xf32> {
    // CHECK-LABEL: func.func @log1p
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[0.000000e+00, 0.69314{{[0-9]*(e\+00)?}}, 2.000000e+00]>
    // CHECK-NOT: "ttir.log1p"
    %0 = "ttir.constant"() <{value = dense<[0.0, 1.0, 6.3890562]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %1 = "ttir.log1p"(%0) : (tensor<3xf32>) -> tensor<3xf32>
    return %1 : tensor<3xf32>
  }

  func.func @log() -> tensor<3xf32> {
    // CHECK-LABEL: func.func @log
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[0.000000e+00, 0.69314{{[0-9]*(e\+00)?}}, 2.000000e+00]>
    // CHECK-NOT: "ttir.log"
    %0 = "ttir.constant"() <{value = dense<[1.0, 2.0, 7.3890562]> : tensor<3xf32>}> : () -> tensor<3xf32>
    %1 = "ttir.log"(%0) : (tensor<3xf32>) -> tensor<3xf32>
    return %1 : tensor<3xf32>
  }

  func.func @logical_not_float() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @logical_not_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]>
    // CHECK-NOT: "ttir.logical_not"
    %0 = "ttir.constant"() <{value = dense<[0.0, 1.0, -2.0, 0.0]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "ttir.logical_not"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @logical_not_sint() -> tensor<4xsi32> {
    // CHECK-LABEL: func.func @logical_not_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1, 0, 0, 1]>
    // CHECK-NOT: "ttir.logical_not"
    %0 = "ttir.constant"() <{value = dense<[0, 1, -2, 0]> : tensor<4xsi32>}> : () -> tensor<4xsi32>
    %1 = "ttir.logical_not"(%0) : (tensor<4xsi32>) -> tensor<4xsi32>
    return %1 : tensor<4xsi32>
  }

  func.func @logical_not_uint() -> tensor<4xui8> {
    // CHECK-LABEL: func.func @logical_not_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1, 0, 0, 1]>
    // CHECK-NOT: "ttir.logical_not"
    %0 = "ttir.constant"() <{value = dense<[0, 1, 255, 0]> : tensor<4xui8>}> : () -> tensor<4xui8>
    %1 = "ttir.logical_not"(%0) : (tensor<4xui8>) -> tensor<4xui8>
    return %1 : tensor<4xui8>
  }

  func.func @reciprocal() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @reciprocal
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, 5.000000e-01, -2.500000e-01, 2.000000e+00]>
    // CHECK-NOT: "ttir.reciprocal"
    %0 = "ttir.constant"() <{value = dense<[1.0, 2.0, -4.0, 0.5]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "ttir.reciprocal"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @reciprocal_no_fold() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @reciprocal_no_fold
    // CHECK: "ttir.zeros"
    // CHECK: "ttir.reciprocal"
    %0 = "ttir.zeros"() <{shape = array<i32: 4>}> : () -> tensor<4xf32>
    %1 = "ttir.reciprocal"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @rsqrt() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @rsqrt
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, 5.000000e-01, 2.500000e-01, 2.000000e+00]>
    // CHECK-NOT: "ttir.rsqrt"
    %0 = "ttir.constant"() <{value = dense<[1.0, 4.0, 16.0, 0.25]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "ttir.rsqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @rsqrt_no_fold() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @rsqrt_no_fold
    // CHECK: "ttir.zeros"
    // CHECK: "ttir.rsqrt"
    %0 = "ttir.zeros"() <{shape = array<i32: 4>}> : () -> tensor<4xf32>
    %1 = "ttir.rsqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @sign_float() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @sign_float
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]>
    // CHECK-NOT: "ttir.sign"
    %0 = "ttir.constant"() <{value = dense<[-4.0, 0.0, 0.5, 3.0]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "ttir.sign"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @sign_bf() -> tensor<4xbf16> {
    // CHECK-LABEL: func.func @sign_bf
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]>
    // CHECK-NOT: "ttir.sign"
    %0 = "ttir.constant"() <{value = dense<[-4.0, 0.0, 0.5, 3.0]> : tensor<4xbf16>}> : () -> tensor<4xbf16>
    %1 = "ttir.sign"(%0) : (tensor<4xbf16>) -> tensor<4xbf16>
    return %1 : tensor<4xbf16>
  }

  func.func @sign_sint() -> tensor<4xsi32> {
    // CHECK-LABEL: func.func @sign_sint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[-1, 0, 1, 1]>
    // CHECK-NOT: "ttir.sign"
    %0 = "ttir.constant"() <{value = dense<[-4, 0, 1, 3]> : tensor<4xsi32>}> : () -> tensor<4xsi32>
    %1 = "ttir.sign"(%0) : (tensor<4xsi32>) -> tensor<4xsi32>
    return %1 : tensor<4xsi32>
  }

  func.func @sign_uint() -> tensor<4xui8> {
    // CHECK-LABEL: func.func @sign_uint
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[0, 0, 1, 1]>
    // CHECK-NOT: "ttir.sign"
    %0 = "ttir.constant"() <{value = dense<[0, 0, 1, 255]> : tensor<4xui8>}> : () -> tensor<4xui8>
    %1 = "ttir.sign"(%0) : (tensor<4xui8>) -> tensor<4xui8>
    return %1 : tensor<4xui8>
  }

  func.func @sin() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @sin
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[0.000000e+00, 5.000000e-01, -5.000000e-01, 1.000000e+00]>
    // CHECK-NOT: "ttir.sin"
    %0 = "ttir.constant"() <{value = dense<[0.0, 0.5235988, -0.5235988, 1.5707963]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "ttir.sin"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @sqrt() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @sqrt
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[1.000000e+00, 2.000000e+00, 4.000000e+00, 5.000000e-01]>
    // CHECK-NOT: "ttir.sqrt"
    %0 = "ttir.constant"() <{value = dense<[1.0, 4.0, 16.0, 0.25]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "ttir.sqrt"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func @tan() -> tensor<4xf32> {
    // CHECK-LABEL: func.func @tan
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<[0.000000e+00, 1.000000e+00, -1.000000e+00, 5.000000e-01]>
    // CHECK-NOT: "ttir.tan"
    %0 = "ttir.constant"() <{value = dense<[0.0, 0.7853982, -0.7853982, 0.4636476]> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "ttir.tan"(%0) : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
