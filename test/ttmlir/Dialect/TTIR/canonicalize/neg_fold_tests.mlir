// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @neg_full() -> tensor<64x64xf32> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 4.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @neg_full
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -4.000000e+00
    %1 = "ttir.neg"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @neg_full_int() -> tensor<64x64xi32> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 2 : i32}> : () -> tensor<64x64xi32>
    // CHECK-LABEL: func.func @neg_full_int
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -2
    %1 = "ttir.neg"(%0) : (tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }

  func.func @neg_full_float_to_int() -> tensor<64x64xi64> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -2.0 : f32}> : () -> tensor<64x64xi64>
    // CHECK-LABEL: func.func @neg_full_float_to_int
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 2 : i32
    %1 = "ttir.neg"(%0) : (tensor<64x64xi64>) -> tensor<64x64xi64>
    return %1 : tensor<64x64xi64>
  }

  func.func @neg_full_int_to_float() -> tensor<64x64xf64> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -5 : i32}> : () -> tensor<64x64xf64>
    // CHECK-LABEL: func.func @neg_full_int_to_float
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 5.000000e+00 : f32
    %1 = "ttir.neg"(%0) : (tensor<64x64xf64>) -> tensor<64x64xf64>
    return %1 : tensor<64x64xf64>
  }

  func.func @neg_full_bf() -> tensor<64x64xbf16> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 4.000000e+00 : f32}> : () -> tensor<64x64xbf16>
    // CHECK-LABEL: func.func @neg_full_bf
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -4.000000e+00
    %1 = "ttir.neg"(%0) : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @neg_full_si64() -> tensor<64x64xsi64> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = 2 : i32}> : () -> tensor<64x64xsi64>
    // CHECK-LABEL: func.func @neg_full_si64
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -2
    %1 = "ttir.neg"(%0) : (tensor<64x64xsi64>) -> tensor<64x64xsi64>
    return %1 : tensor<64x64xsi64>
  }

  func.func @neg_zeros_float() -> tensor<64x64xf32> {
    %0 = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @neg_zeros_float
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -0.000000e+00
    %1 = "ttir.neg"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @neg_zeros_int() -> tensor<64x64xi32> {
    %0 = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xi32>
    // CHECK-LABEL: func.func @neg_zeros_int
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.zeros"
    %1 = "ttir.neg"(%0) : (tensor<64x64xi32>) -> tensor<64x64xi32>
    return %1 : tensor<64x64xi32>
  }

  func.func @neg_minus_ones() -> tensor<64x64xf32> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -1.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @neg_minus_ones
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.ones"
    %1 = "ttir.neg"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @neg_ones() -> tensor<64x64xf32> {
    %0 = "ttir.ones"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @neg_ones
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -1.000000e+00
    %1 = "ttir.neg"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }

  func.func @neg_constant() -> tensor<2x2xf32> {
    %0 = "ttir.constant"() {
      value = dense<[[-1.000000e+00, 2.000000e+00], [3.000000e+00, -4.000000e+00]]> : tensor<2x2xf32>
    } : () -> tensor<2x2xf32>
    // CHECK-LABEL: func.func @neg_constant
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}1.000000e+00, -2.000000e+00], {{\[}}-3.000000e+00, 4.000000e+00]]>
    %1 = "ttir.neg"(%0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
