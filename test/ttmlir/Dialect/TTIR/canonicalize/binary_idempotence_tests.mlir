// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @binary_idempotence(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK-NOT: "ttir.logical_and"
    %0 = ttir.empty() : tensor<64x64xf32>
    %1 = "ttir.logical_and"(%arg0, %arg0, %0) : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
}
