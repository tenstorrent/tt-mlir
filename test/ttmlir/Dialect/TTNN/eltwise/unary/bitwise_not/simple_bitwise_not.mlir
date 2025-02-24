// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func @bitwise_not(%arg0: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = tensor.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_not"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_not"
    // CHECK-SAME: tensor<64x128xui32
    // CHECK-SAME: -> tensor<64x128xui32
    return %1 : tensor<64x128xi32>
  }
}
