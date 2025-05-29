// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    // CHECK: = "ttnn.neg"
    // CHECK: = "ttnn.add"
    // CHECK-NOT: = "ttnn.subtract"
    %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<1x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
