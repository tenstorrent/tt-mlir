// RUN: not ttmlir-opt --ttir-to-ttnn-backend-pipeline --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for reduce(prod) op

// CHECK: error: 'ttnn.prod' op TTNN only supports reduce(prod) along one dimension or all dimensions.
func.func public @test_reduce_prod_multiple_dims(%arg0: tensor<128x10x32x4xf32>) -> tensor<128x32xf32> {
  %0 = tensor.empty() : tensor<128x32xf32>
  %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1 : i32, 3 : i32], keep_dim = false}> : (tensor<128x10x32x4xf32>, tensor<128x32xf32>) -> tensor<128x32xf32>
  return %1 : tensor<128x32xf32>
}

// -----
// CHECK: error: 'ttnn.prod' op TTNN only supports Reduce(prod) along all dimensions for bfloat16 datatype.
func.func public @test_reduce_prod_all_dims_f32(%arg0: tensor<128x10x32x4xf32>) -> tensor<1xf32> {
  %0 = tensor.empty() : tensor<1xf32>
  %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [0: i32, 1 : i32, 2: i32, 3 : i32], keep_dim = false}> : (tensor<128x10x32x4xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

// -----
// CHECK: error: 'ttnn.prod' op Input tensor rank is greater than 4 for reduce(product).
func.func public @test_reduce_prod_higher_rank(%arg0: tensor<128x10x32x4x1xf32>) -> tensor<10x32x4x1xf32> {
  %0 = tensor.empty() : tensor<10x32x4x1xf32>
  %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<128x10x32x4x1xf32>, tensor<10x32x4x1xf32>) -> tensor<10x32x4x1xf32>
  return %1 : tensor<10x32x4x1xf32>
}
