// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-const-eval=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @forward(%arg0: tensor<1x18xi32>, %arg1: tensor<1x18xi32>) -> tensor<1x18xui32> {
  %0 = "ttir.constant"() <{value = dense<5> : tensor<1x18xui32>}> : () -> tensor<1x18xui32>
  %1 = "ttir.constant"() <{value = dense<1> : tensor<1x18xui32>}> : () -> tensor<1x18xui32>
  %2 = "ttir.logical_left_shift"(%0, %1) : (tensor<1x18xui32>, tensor<1x18xui32>) -> tensor<1x18xui32>
  // CHECK: "ttnn.logical_left_shift"
  // CHECK-SAME: tensor<1x18xui32
  // CHECK-SAME: tensor<1x18xui32
  // CHECK-SAME: -> tensor<1x18xui32
  return %2 : tensor<1x18xui32>
}
