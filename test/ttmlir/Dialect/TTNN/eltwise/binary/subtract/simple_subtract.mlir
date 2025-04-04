// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.subtract"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: -> tensor<64x128xbf16
    return %1 : tensor<64x128xbf16>
  }

  func.func @subtract_with_neg_and_add_1(%arg0: tensor<64x128xbf16>, %arg1: tensor<1x128xbf16>) -> tensor<64x128xbf16> {
    %0 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: = "ttnn.neg"
    // CHECK: = "ttnn.add"
    // CHECK-NOT: = "ttnn.subtract"
    %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<1x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }

  func.func @subtract_with_neg_and_add_2(%arg0: tensor<1x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
     %0 = ttir.empty() : tensor<64x128xbf16>
    // CHECK: = "ttnn.neg"
    // CHECK: = "ttnn.add"
    // CHECK-NOT: = "ttnn.subtract"
    %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
