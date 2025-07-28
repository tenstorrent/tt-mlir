// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @reshape_positive(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
    %0 = ttir.empty() : tensor<2x4x32x32xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 32: i32]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: <{shape = [2 : i32, 4 : i32, 32 : i32, 32 : i32]}>
    // CHECK-SAME: tensor<4x2x32x32xbf16
    // CHECK-SAME: -> tensor<2x4x32x32xbf16
    return %1 : tensor<2x4x32x32xbf16>
  }

  func.func @reshape_with_minus_one(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
    %0 = ttir.empty() : tensor<2x4x32x32xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, -1: i32, 32: i32, 32: i32]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: <{shape = [2 : i32, -1 : i32, 32 : i32, 32 : i32]}>
    // CHECK-SAME: tensor<4x2x32x32xbf16
    // CHECK-SAME: -> tensor<2x4x32x32xbf16
    return %1 : tensor<2x4x32x32xbf16>
  }
}
