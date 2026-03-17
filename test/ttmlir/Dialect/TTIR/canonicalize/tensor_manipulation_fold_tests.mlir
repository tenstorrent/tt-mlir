// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @constant_fold_splat() -> tensor<8x6x6xi32> {
    // CHECK-LABEL: func.func @constant_fold_splat
    // CHECK-NOT: "ttir.broadcast"
    // CHECK-NOT: "ttir.repeat"
    // CHECK-NOT: "ttir.repeat_interleave"
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 4
    // CHECK-SAME: shape = array<i32: 8, 6, 6>
    %0 = "ttir.full"() <{fill_value = 4 : i32, shape = array<i32: 1, 1>}> : () -> tensor<1x1xi32>
    %1 = "ttir.broadcast"(%0) {broadcast_dimensions = array<i64: 2, 8>} : (tensor<1x1xi32>) -> tensor<2x8xi32>
    %2 = "ttir.repeat"(%1) {repeat_dimensions = array<i64: 3, 2>} : (tensor<2x8xi32>) -> tensor<6x16xi32>
    %3 = "ttir.repeat_interleave"(%2) <{repeats = 3 : ui32, dim = 1 : si32}> : (tensor<6x16xi32>) -> tensor<6x48xi32>
    %4 = "ttir.reshape"(%3) {shape = [8 : i32, 6 : i32, 6 : i32]} : (tensor<6x48xi32>) -> tensor<8x6x6xi32>
    return %4 : tensor<8x6x6xi32>
  }

  func.func @constant_fold_reshape() -> tensor<10000x1000000x2000xbf16> {
    // CHECK-LABEL: func.func @constant_fold_reshape
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.zeros"
    // CHECK-SAME: shape = array<i32: 10000, 1000000, 2000>
    %0 = "ttir.zeros"() <{shape = array<i32: 2000000, 10000000>}> : () -> tensor<2000000x10000000xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [10000 : i32, 1000000 : i32, 2000 : i32]}> : (tensor<2000000x10000000xbf16>) -> tensor<10000x1000000x2000xbf16>
    return %1 : tensor<10000x1000000x2000xbf16>
  }
}
