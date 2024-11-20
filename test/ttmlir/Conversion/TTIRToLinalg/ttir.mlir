// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s
module attributes{} {
  func.func @add(
    %arg0: tensor<32x32xf32>,  // First input tensor
    %arg1: tensor<32x32xf32>,  // Second input tensor
    %arg2: tensor<32x32xf32>   // Output tensor (result stored here)
  ) -> tensor<32x32xf32> {
    %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
     // CHECK: [[VAL1:%[0-9]+]] = linalg.add ins(%arg{{[0-9]+}}, %arg{{[0-9]+}} : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg{{[0-9]+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
