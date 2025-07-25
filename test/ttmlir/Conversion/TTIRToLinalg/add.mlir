// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

module attributes {} {
  func.func @test_add(%arg0: tensor<1x784xf32>, %arg1: tensor<784xf32>) -> tensor<1x784xf32> {
    // CHECK: = tensor.empty() : [[SIZE:tensor<1x784xf32>]]
    %0 = ttir.empty() : tensor<1x784xf32>
    // CHECK: [[BROADCASTED: %.+]] = linalg.broadcast
    // CHECK: [[RESULT: %[0-9]+]] = linalg.add
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784xf32>, tensor<1x784xf32>) -> tensor<1x784xf32>
    // CHECK: return[[RESULT]] : [[SIZE]]
    return %1 : tensor<1x784xf32>
  }
}
