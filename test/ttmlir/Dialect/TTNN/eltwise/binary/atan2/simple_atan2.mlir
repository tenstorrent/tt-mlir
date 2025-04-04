// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @atan2(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.atan2"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: "ttnn.atan2"
    // CHECK-SAME: tensor<32x32xbf16
    // CHECK-SAME: tensor<32x32xbf16
    // CHECK-SAME: -> tensor<32x32xbf16
    return %1 : tensor<32x32xbf16>
  }
}
