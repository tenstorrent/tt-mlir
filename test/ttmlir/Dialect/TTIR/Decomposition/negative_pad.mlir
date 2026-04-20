// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

// Negative-only padding — should lower to slice, no pad
module {
  func.func @pad_negative(%arg0: tensor<1x1x5x5xbf16>) -> tensor<1x1x3x3xbf16> {
    // CHECK: [[VAL:%.+]] = "ttir.slice_static"
    // CHECK: begins = [0 : i32, 0 : i32, 1 : i32, 1 : i32]
    // CHECK: ends = [1 : i32, 1 : i32, 4 : i32, 4 : i32]
    // CHECK: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, -1, -1, -1, -1>, value = 0.0 : f32}> : (tensor<1x1x5x5xbf16>) -> tensor<1x1x3x3xbf16>
    return %1 : tensor<1x1x3x3xbf16>
  }
}

// Mixed positive and negative padding — should lower to slice then pad.
module {
  func.func @pad_mixed(%arg0: tensor<1x1x5x5xbf16>) -> tensor<1x1x7x3xbf16> {
    // CHECK: [[VAL:%.+]] = "ttir.slice_static"
    // CHECK: begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32]
    // CHECK: ends = [1 : i32, 1 : i32, 5 : i32, 4 : i32]
    // CHECK: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    // CHECK: [[VAL2:%.+]] = "ttir.pad"
    // CHECK: padding = array<i32: 0, 0, 0, 0, 1, 1, 0, 0>
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 1, 1, -1, -1>, value = 0.0 : f32}> : (tensor<1x1x5x5xbf16>) -> tensor<1x1x7x3xbf16>
    return %1 : tensor<1x1x7x3xbf16>
  }
}
