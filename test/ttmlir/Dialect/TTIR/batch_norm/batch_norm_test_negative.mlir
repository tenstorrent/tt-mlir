// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Verify that the parsing fails if tensors don't have four dimensions
module attributes {} {
  func.func public @test_batch_norm_1(%arg0: tensor<2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2x2xf32> {
    %0 = ttir.empty() : tensor<2x2x2xf32>
    %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 1 : i32, epsilon = 0.000000e+00 : f32, training = false}> : (tensor<2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
    // CHECK: error: 'ttir.batch_norm' op input tensor must be a 4D tensor
    return %1 : tensor<2x2x2xf32>
  }
}
