// RUN: ttmlir-opt --ttir-constant-as-fill %s | FileCheck %s
func.func public @add5(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: = tensor.empty
  // CHECK: = "ttir.fill"
  %0 = "ttir.constant"() <{value = dense<5.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  // CHECK: = tensor.empty
  %1 = tensor.empty() : tensor<32x32xf32>
  %2 = "ttir.add"(%arg0, %0, %1) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
