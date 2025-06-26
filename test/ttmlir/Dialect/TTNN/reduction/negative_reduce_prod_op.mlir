// RUN: not ttmlir-opt --ttir-to-ttnn-backend-pipeline --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for reduce(prod) op

// CHECK: error: 'ttnn.prod' op Input tensor rank is greater than 4 for reduce(product).
func.func public @test_reduce_prod_higher_rank(%arg0: tensor<128x10x32x4x1xf32>) -> tensor<10x32x4x1xf32> {
  %0 = ttir.empty() : tensor<10x32x4x1xf32>
  %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<128x10x32x4x1xf32>, tensor<10x32x4x1xf32>) -> tensor<10x32x4x1xf32>
  return %1 : tensor<10x32x4x1xf32>
}
