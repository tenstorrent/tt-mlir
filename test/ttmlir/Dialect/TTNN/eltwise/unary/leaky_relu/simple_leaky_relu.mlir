// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @leaky_relu(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = "ttir.leaky_relu"(%arg0, %0) <{parameter = 0.01 : f32, operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %[[RETURN_VALUE:[0-9]+]] = "ttnn.leaky_relu"(%arg0)
    // CHECK-SAME: (tensor<64x128xf32, {{.*}}>)
    // CHECK-SAME: -> tensor<64x128xf32, {{.*}}>
    return %1 : tensor<64x128xf32>
    // CHECK: return %[[RETURN_VALUE]] : tensor<64x128xf32, {{.*}}>
  }
}
