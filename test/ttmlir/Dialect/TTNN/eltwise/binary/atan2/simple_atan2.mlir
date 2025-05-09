// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @atan2(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.atan2"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: "ttnn.atan2"
    // CHECK-SAME: tensor<32x32xf32
    // CHECK-SAME: tensor<32x32xf32
    // CHECK-SAME: -> tensor<32x32xf32
    return %1 : tensor<32x32xf32>
  }
}
