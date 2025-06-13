// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: "ttnn.subtract"
  // CHECK-SAME: tensor<64x128xf32
  // CHECK-SAME: tensor<64x128xf32
  // CHECK-SAME: -> tensor<64x128xf32
  return %1 : tensor<64x128xf32>
}
