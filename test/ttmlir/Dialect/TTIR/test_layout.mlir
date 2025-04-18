// RUN: ttmlir-opt --tt-register-device --ttir-layout %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<8x64x128xf32>, %arg1: tensor<8x64x128xf32>) -> tensor<8x64x128xf32> {
    // CHECK: = ttir.empty() : tensor<8x64x128xf32, #layout>
    %0 = ttir.empty() : tensor<8x64x128xf32>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<8x64x128xf32>, tensor<8x64x128xf32>, tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    return %1 : tensor<8x64x128xf32>
  }

  func.func @test_unused_argument(%arg0: tensor<8x64x128xf32>) -> tensor<8x64x128xf32> {
    // CHECK: = ttir.empty() : tensor<8x64x128xf32, #layout>
    %0 = ttir.empty() : tensor<8x64x128xf32>
    return %0 : tensor<8x64x128xf32>
  }
}
