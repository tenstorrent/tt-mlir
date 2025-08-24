// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @remainder(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.remainder"(%arg0, %arg1, %0) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: "ttnn.remainder"
    // CHECK-SAME: tensor<32x32xf32
    // CHECK-SAME: tensor<32x32xf32
    // CHECK-SAME: -> tensor<32x32xf32
    return %1 : tensor<32x32xf32>
  }
}
