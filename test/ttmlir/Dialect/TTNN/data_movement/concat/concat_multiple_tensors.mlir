// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32x128xf32>) -> tensor<32x224xf32> {
    %0 = ttir.empty() : tensor<32x224xf32>
    // CHECK: = "ttnn.concat"
    %1 = "ttir.concat"(%arg0, %arg1, %arg2, %0) <{dim = 1 : si32}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x128xf32>, tensor<32x224xf32>) -> tensor<32x224xf32>
    return %1 : tensor<32x224xf32>
  }
}
