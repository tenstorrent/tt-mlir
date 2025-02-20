// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = "ttir.subtract"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %[[RETURN_VALUE:[0-9]+]] = "ttnn.subtract"(%arg0, %arg1)
    // CHECK-SAME: (tensor<64x128xf32, {{.*}}>, tensor<64x128xf32, {{.*}}>)
    // CHECK-SAME: -> tensor<64x128xf32, {{.*}}>
    return %1 : tensor<64x128xf32>
    // CHECK: return %[[RETURN_VALUE]] : tensor<64x128xf32, {{.*}}>
  }
}
