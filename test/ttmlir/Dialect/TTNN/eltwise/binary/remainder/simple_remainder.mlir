// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @remainder(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = tensor.empty() : tensor<32x32xf32>
    %1 = "ttir.remainder"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: %[[RETURN_VALUE:[0-9]+]] = "ttnn.remainder"(%arg0, %arg1)
    // CHECK-SAME: (tensor<32x32xf32, {{.*}}>, tensor<32x32xf32, {{.*}}>)
    // CHECK-SAME: -> tensor<32x32xf32, {{.*}}>
    return %1 : tensor<32x32xf32>
    // CHECK: return %[[RETURN_VALUE]] : tensor<32x32xf32, {{.*}}>
  }
}
