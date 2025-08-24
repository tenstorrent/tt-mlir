// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward() -> tensor<32x224xf32> {
    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = ttir.empty() : tensor<32x64xf32>
    %2 = ttir.empty() : tensor<32x128xf32>
    %3 = ttir.empty() : tensor<32x224xf32>
    // CHECK: = "ttnn.concat"
    %4 = "ttir.concat"(%0, %1, %2, %3) <{dim = 1 : si32}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x128xf32>, tensor<32x224xf32>) -> tensor<32x224xf32>
    return %4 : tensor<32x224xf32>
  }
}
