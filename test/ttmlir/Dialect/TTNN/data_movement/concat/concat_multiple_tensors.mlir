// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward() -> tensor<32x224xf32> {
    // CHECK: = "ttnn.empty"
    %0 = tensor.empty() : tensor<32x32xf32>
    // CHECK: = "ttnn.empty"
    %1 = tensor.empty() : tensor<32x64xf32>
    // CHECK: = "ttnn.empty"
    %2 = tensor.empty() : tensor<32x128xf32>
    // CHECK: = "ttnn.empty"
    %3 = tensor.empty() : tensor<32x224xf32>
    // CHECK: = "ttnn.concat"
    %4 = "ttir.concat"(%0, %1, %2, %3) <{dim = 1 : si32}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x128xf32>, tensor<32x224xf32>) -> tensor<32x224xf32>
    return %4 : tensor<32x224xf32>
  }
}
