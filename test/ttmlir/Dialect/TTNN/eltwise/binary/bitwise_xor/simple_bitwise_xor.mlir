// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module attributes {} {
  func.func @bitwise_xor(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = tensor.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    // CHECK: %[[RETURN_VALUE:[0-9]+]] = "ttnn.bitwise_xor"(%arg0, %arg1)
    // CHECK-SAME: (tensor<64x128xui32, {{.*}}>, tensor<64x128xui32, {{.*}}>)
    // CHECK-SAME: -> tensor<64x128xui32, {{.*}}>
    return %1 : tensor<64x128xi32>
    // CHECK: return %[[RETURN_VALUE]] : tensor<64x128xui32, {{.*}}>
  }
}
