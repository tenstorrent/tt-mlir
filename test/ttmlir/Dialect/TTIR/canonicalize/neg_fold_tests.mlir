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
}
