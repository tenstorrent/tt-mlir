// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @forward(%arg0: tensor<1x18xi32>, %arg1: tensor<1x18xi32>) -> tensor<1x18xui32> {
  %0 = "ttir.constant"() <{value = dense<5> : tensor<1x18xui32>}> : () -> tensor<1x18xui32>
  %1 = "ttir.constant"() <{value = dense<1> : tensor<1x18xui32>}> : () -> tensor<1x18xui32>
  %2 = ttir.empty() : tensor<1x18xui32>
  %3 = "ttir.logical_right_shift"(%0, %1, %2) : (tensor<1x18xui32>, tensor<1x18xui32>, tensor<1x18xui32>) -> tensor<1x18xui32>
  // CHECK: "ttnn.logical_right_shift"
  // CHECK-SAME: tensor<1x18xui32
  // CHECK-SAME: tensor<1x18xui32
  // CHECK-SAME: -> tensor<1x18xui32
  return %3 : tensor<1x18xui32>
}
