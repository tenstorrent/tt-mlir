// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @test_matmul(%arg0: tensor<1x784xf32>, %arg1: tensor<784x512xf32>) -> tensor<1x512xf32> {
    // CHECK: = tensor.empty() : [[SIZE:tensor<1x512xf32>]]
    %0 = ttir.empty() : tensor<1x512xf32>
    // CHECK: = tosa.reshape %arg0
    // CHECK: = tosa.reshape %arg1
    // CHECK: = tosa.matmul
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    // CHECK: [[RESULT: %[0-9]+]] = tosa.reshape
    // CHECK: return[[RESULT]] : [[SIZE]]
    return %1 : tensor<1x512xf32>
  }
}
