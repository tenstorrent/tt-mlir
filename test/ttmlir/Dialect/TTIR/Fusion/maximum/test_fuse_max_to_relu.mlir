// RUN: ttmlir-opt --ttir-fusion %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %zero = "ttir.constant"() {value = dense<0.000000e+00> : tensor<64x128xf32>} : () -> tensor<64x128xf32>
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: "ttir.relu"
    %1 = "ttir.maximum"(%arg0, %zero, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
