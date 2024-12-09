// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @binary_idempotence(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK-NOT: "ttir.logical_and"
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = "ttir.logical_and"(%arg0, %arg0, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
}
