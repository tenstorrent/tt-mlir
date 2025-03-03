// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func @bitwise_or(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = tensor.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_or"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: "ttnn.bitwise_or"
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: tensor<64x128xsi32
    // CHECK-SAME: -> tensor<64x128xsi32
    return %1 : tensor<64x128xi32>
  }
}
