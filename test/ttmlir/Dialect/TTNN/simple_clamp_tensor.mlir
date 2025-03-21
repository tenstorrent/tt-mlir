// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func public @test_clamp_tensor(%arg0: tensor<32x64xf32>, %arg1: tensor<32x64xf32>, %arg2: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = tensor.empty() : tensor<32x64xf32>
    // CHECK: %{{[0-9]+}} = "ttnn.clamp_tensor"(%arg0, %arg1, %arg2)
    // CHECK-SAME: tensor<32x64xf32,
    // CHECK-SAME: tensor<32x64xf32,
    // CHECK-SAME: tensor<32x64xf32,
    // CHECK-SAME: -> tensor<32x64xf32,
    %1 = "ttir.clamp_tensor"(%arg0, %arg1, %arg2, %0) : (tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }
}
