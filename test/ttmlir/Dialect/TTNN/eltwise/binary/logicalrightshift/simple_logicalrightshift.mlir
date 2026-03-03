// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @forward(%arg0: tensor<1x18xui32>, %arg1: tensor<1x18xui32>) -> tensor<1x18xui32> {
  %2 = "ttir.logical_right_shift"(%arg0, %arg1) : (tensor<1x18xui32>, tensor<1x18xui32>) -> tensor<1x18xui32>
  // CHECK: "ttnn.logical_right_shift"
  // CHECK-SAME: tensor<1x18xui32
  // CHECK-SAME: tensor<1x18xui32
  // CHECK-SAME: -> tensor<1x18xui32
  return %2 : tensor<1x18xui32>
}
