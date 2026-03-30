// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @constant_fold_splat() -> tensor<1x2x2xi32> {
    // CHECK-LABEL: func.func @constant_fold_splat
    // CHECK-NOT: "ttir.broadcast"
    // CHECK-NOT: "ttir.repeat"
    // CHECK-NOT: "ttir.repeat_interleave"
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 4
    // CHECK-SAME: shape = array<i32: 1, 2, 2>
    %0 = "ttir.full"() <{fill_value = 4 : i32, shape = array<i32: 1, 1>}> : () -> tensor<1x1xi32>
    %1 = "ttir.broadcast"(%0) {broadcast_dimensions = array<i64: 2, 8>} : (tensor<1x1xi32>) -> tensor<2x8xi32>
    %2 = "ttir.repeat"(%1) {repeat_dimensions = array<i64: 3, 2>} : (tensor<2x8xi32>) -> tensor<6x16xi32>
    %3 = "ttir.repeat_interleave"(%2) <{repeats = 3 : ui32, dim = 1 : si32}> : (tensor<6x16xi32>) -> tensor<6x48xi32>
    %4 = "ttir.reshape"(%3) {shape = [8 : i32, 6 : i32, 6 : i32]} : (tensor<6x48xi32>) -> tensor<8x6x6xi32>
    %5 = "ttir.permute"(%4) {permutation = array<i64: 1, 0, 2>} : (tensor<8x6x6xi32>) -> tensor<6x8x6xi32>
    %6 = "ttir.slice_static"(%5) {begins = [1 : i32, 1 : i32, 4 : i32], ends = [2 : i32, 4 : i32, 2 : i32], step = [1 : i32, 2 : i32, -1 : i32]} : (tensor<6x8x6xi32>) -> tensor<1x2x2xi32>
    return %6 : tensor<1x2x2xi32>
  }

  func.func @constant_fold_float() -> tensor<3x4xbf16> {
    // CHECK-LABEL: func.func @constant_fold_float
    // CHECK-NOT: "ttir.permute"
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}1.000000e+00, 3.000000e+00, 5.000000e+00, 7.000000e+00],
    %0 = "ttir.constant"() {value = dense<[[[1., 2.], [3., 4.], [5., 6.]], [[7., 8.], [9., 10.], [11., 12.]]]> : tensor<2x3x2xbf16>} : () -> tensor<2x3x2xbf16>
    %1 = "ttir.permute"(%0) {permutation = array<i64: 2, 0, 1>} : (tensor<2x3x2xbf16>) -> tensor<2x2x3xbf16>
    %2 = "ttir.reshape"(%1) {shape = [3 : i32, 4 : i32]} : (tensor<2x2x3xbf16>) -> tensor<3x4xbf16>
    return %2 : tensor<3x4xbf16>
  }

  func.func @constant_fold_int() -> tensor<3x4xui64> {
    // CHECK-LABEL: func.func @constant_fold_int
    // CHECK-NOT: "ttir.permute"
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}1, 3, 5, 7],
    %0 = "ttir.constant"() {value = dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xui64>} : () -> tensor<2x3x2xui64>
    %1 = "ttir.permute"(%0) {permutation = array<i64: 2, 0, 1>} : (tensor<2x3x2xui64>) -> tensor<2x2x3xui64>
    %2 = "ttir.reshape"(%1) {shape = [3 : i32, 4 : i32]} : (tensor<2x2x3xui64>) -> tensor<3x4xui64>
    return %2 : tensor<3x4xui64>
  }

  func.func @repeat_interleave_identity_fold() -> tensor<32x32xi32> {
    // CHECK-LABEL: func.func @repeat_interleave_identity_fold
    // CHECK: "ttir.zeros"
    // CHECK-NOT: "ttir.repeat_interleave"
    // CHECK: return %0
    %0 = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xi32>
    %1 = "ttir.repeat_interleave"(%0) <{repeats = 1 : ui32, dim = 1 : si32}> : (tensor<32x32xi32>) -> tensor<32x32xi32>
    %2 = "ttir.repeat_interleave"(%1) <{repeats = 1 : ui32, dim = 0 : si32}> : (tensor<32x32xi32>) -> tensor<32x32xi32>
    return %2 : tensor<32x32xi32>
  }

  func.func @slice_fold_int() -> tensor<2x2xui64> {
    // CHECK-LABEL: func.func @slice_fold_int
    // CHECK-NOT: "ttir.slice_static"
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}2, 6], {{\[}}8, 12]]> : tensor<2x2xui64>
    %0 = "ttir.constant"() {value = dense<[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]> : tensor<2x3x2xui64>} : () -> tensor<2x3x2xui64>
    %1 = "ttir.slice_static"(%0) {begins = [0 : i32, 0 : i32, -1 : i32], ends = [2 : i32, 3 : i32, 0 : i32], step = [1 : i32, 2 : i32, -1 : i32]} : (tensor<2x3x2xui64>) -> tensor<2x2x1xui64>
    %2 = "ttir.reshape"(%1) {shape = [2 : i32, 2 : i32]} : (tensor<2x2x1xui64>) -> tensor<2x2xui64>
    return %2 : tensor<2x2xui64>
  }

  func.func @slice_fold_float() -> tensor<2x2xbf16> {
    // CHECK-LABEL: func.func @slice_fold_float
    // CHECK-NOT: "ttir.slice_static"
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<{{\[\[}}2.000000e+00, 6.000000e+00], {{\[}}8.000000e+00, 1.200000e+01]]> : tensor<2x2xbf16>
    %0 = "ttir.constant"() {value = dense<[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]> : tensor<2x3x2xbf16>} : () -> tensor<2x3x2xbf16>
    %1 = "ttir.slice_static"(%0) {begins = [0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 3 : i32, 0 : i32], step = [1 : i32, 2 : i32, -1 : i32]} : (tensor<2x3x2xbf16>) -> tensor<2x2x1xbf16>
    %2 = "ttir.reshape"(%1) {shape = [2 : i32, 2 : i32]} : (tensor<2x2x1xbf16>) -> tensor<2x2xbf16>
    return %2 : tensor<2x2xbf16>
  }
}
