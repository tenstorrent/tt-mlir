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

  func.func @neg_full_negative() -> tensor<64x64xf32> {
    %0 = "ttir.full"() <{shape = array<i32: 64, 64>, fill_value = -5.000000e+00 : f32}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @neg_full_negative
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 5.000000e+00
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

  func.func @neg_zeros() -> tensor<64x64xf32> {
    %0 = "ttir.zeros"() <{shape = array<i32: 64, 64>}> : () -> tensor<64x64xf32>
    // CHECK-LABEL: func.func @neg_zeros
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.zeros"
    %1 = "ttir.neg"(%0) : (tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
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

  func.func @neg_broadcast_reshape() -> tensor<128x64xf32> {
    %one = "ttir.full"() <{shape = array<i32: 1, 1>, fill_value = 1.000000e+00 : f32}> : () -> tensor<1x1xf32>
    %reshaped = "ttir.reshape"(%one) <{shape = [1 : i32, 1 : i32]}> : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %broadcasted = "ttir.broadcast"(%reshaped) <{broadcast_dimensions = array<i64: 64, 64>}> : (tensor<1x1xf32>) -> tensor<64x64xf32>
    %repeated = "ttir.repeat_interleave"(%broadcasted) {repeats = 2 : ui32, dim = 0 : si32} : (tensor<64x64xf32>) -> tensor<128x64xf32>
    // CHECK-LABEL: func.func @neg_broadcast_reshape
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = -1.000000e+00
    // CHECK-SAME: shape = array<i32: 128, 64>
    %neg = "ttir.neg"(%repeated) : (tensor<128x64xf32>) -> tensor<128x64xf32>
    return %neg : tensor<128x64xf32>
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

  func.func @neg_constant_reshape() -> tensor<3x2xi32> {
    %0 = "ttir.constant"() {
      value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
    } : () -> tensor<2x3xi32>
    %1 = "ttir.reshape"(%0) <{shape = [3 : i32, 2 : i32]}> : (tensor<2x3xi32>) -> tensor<3x2xi32>
    // CHECK-LABEL: func.func @neg_constant
    // CHECK-NOT: "ttir.neg"
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}-1, -2], {{\[}}-3, -4], {{\[}}-5, -6]]>
    %2 = "ttir.neg"(%1) : (tensor<3x2xi32>) -> tensor<3x2xi32>
    return %2 : tensor<3x2xi32>
  }
}
