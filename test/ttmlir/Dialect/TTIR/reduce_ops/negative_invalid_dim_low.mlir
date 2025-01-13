// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for reduce ops

// CHECK: error: 'ttir.sum' op Reduce dimensions are out of range
func.func public @test_reduce_add_invalid_dim_low(%arg0: tensor<128x10xf32>, %arg1: tensor<1xf32>) -> tensor<128xf32> {
  %0 = tensor.empty() : tensor<128xf32>
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-3 : i32], keep_dim = false}> : (tensor<128x10xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %1 : tensor<128xf32>
}
