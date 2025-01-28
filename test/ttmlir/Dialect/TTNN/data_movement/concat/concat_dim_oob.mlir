// RUN: not ttmlir-opt --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s
// CHECK: error: 'ttir.concat' op Invalid dimension 2 for concatenation.
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
    %0 = tensor.empty() : tensor<32x96xf32>
    %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 2 : si32}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
    return %1 : tensor<32x96xf32>
  }
}
