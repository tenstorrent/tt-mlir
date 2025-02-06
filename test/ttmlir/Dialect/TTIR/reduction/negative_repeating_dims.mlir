// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for reduce ops

// CHECK: error: 'ttir.sum' op Reduce dimensions are not unique
func.func public @test_reduce_add_repeating_dims(%arg0: tensor<128x10x32x4xf32>, %arg1: tensor<1xf32>) -> tensor<128xf32> {
  %0 = tensor.empty() : tensor<128xf32>
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [1 : i32, 2 : i32, 3 : i32, 2 : i32], keep_dim = false}> : (tensor<128x10x32x4xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}
