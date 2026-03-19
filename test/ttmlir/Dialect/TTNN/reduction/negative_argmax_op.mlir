// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for argmax op

// CHECK: error: 'ttnn.argmax' op dim attribute value 3 is out of range for input tensor of rank 2, expected value in range [-2, 1]
func.func @test_argmax_dim_out_of_range_positive(%arg0: tensor<64x64xf32>) -> tensor<64xui32> {
  %0 = "ttnn.argmax"(%arg0) <{dim = 3 : i32, keep_dim = false, use_multicore = false}> : (tensor<64x64xf32>) -> tensor<64xui32>
  return %0 : tensor<64xui32>
}

// -----

// CHECK: error: 'ttnn.argmax' op dim attribute value -3 is out of range for input tensor of rank 2, expected value in range [-2, 1]
func.func @test_argmax_dim_out_of_range_negative(%arg0: tensor<64x64xf32>) -> tensor<64xui32> {
  %0 = "ttnn.argmax"(%arg0) <{dim = -3 : i32, keep_dim = false, use_multicore = false}> : (tensor<64x64xf32>) -> tensor<64xui32>
  return %0 : tensor<64xui32>
}

// -----

// CHECK: error: 'ttnn.argmax' op expected output shape (64), got (64, 64)
func.func @test_argmax_wrong_output_shape(%arg0: tensor<64x64xf32>) -> tensor<64x64xui32> {
  %0 = "ttnn.argmax"(%arg0) <{dim = 1 : i32, keep_dim = false, use_multicore = false}> : (tensor<64x64xf32>) -> tensor<64x64xui32>
  return %0 : tensor<64x64xui32>
}

// -----

// CHECK: error: 'ttnn.argmax' op expected output shape (64, 1), got (64, 64)
func.func @test_argmax_wrong_output_shape_keep_dim(%arg0: tensor<64x64xf32>) -> tensor<64x64xui32> {
  %0 = "ttnn.argmax"(%arg0) <{dim = 1 : i32, keep_dim = true, use_multicore = false}> : (tensor<64x64xf32>) -> tensor<64x64xui32>
  return %0 : tensor<64x64xui32>
}

// -----

// CHECK: error: 'ttnn.argmax' op expected output shape (), got (64)
func.func @test_argmax_no_dim_wrong_output_shape(%arg0: tensor<64x64xf32>) -> tensor<64xui32> {
  %0 = "ttnn.argmax"(%arg0) <{keep_dim = false, use_multicore = false}> : (tensor<64x64xf32>) -> tensor<64xui32>
  return %0 : tensor<64xui32>
}
