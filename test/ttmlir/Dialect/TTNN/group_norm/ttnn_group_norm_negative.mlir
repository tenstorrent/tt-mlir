// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for TTNN GroupNormOp

// CHECK: error: 'ttnn.group_norm' op input and output must have the same shape
func.func @group_norm_shape_mismatch(%arg0: tensor<2x1x16x32xf32>) -> tensor<2x1x16x64xf32> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x64xf32>
  return %0 : tensor<2x1x16x64xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op input tensor must be 4D with shape [N, 1, H×W, C]
func.func @group_norm_wrong_rank(%arg0: tensor<2x32xf32>) -> tensor<2x32xf32> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x32xf32>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op input tensor dimension 1 must be exactly 1
func.func @group_norm_wrong_dim1(%arg0: tensor<2x2x16x32xf32>) -> tensor<2x2x16x32xf32> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x2x16x32xf32>) -> tensor<2x2x16x32xf32>
  return %0 : tensor<2x2x16x32xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op N × H×W must be a multiple of 32 (tile size), got 30
func.func @group_norm_not_tile_aligned(%arg0: tensor<2x1x15x32xf32>) -> tensor<2x1x15x32xf32> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x1x15x32xf32>) -> tensor<2x1x15x32xf32>
  return %0 : tensor<2x1x15x32xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op number of channels (32) must be divisible by num_groups (7)
func.func @group_norm_channels_not_divisible(%arg0: tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 7 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32>
  return %0 : tensor<2x1x16x32xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op num_groups must be positive, got 0
func.func @group_norm_zero_groups(%arg0: tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 0 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x1x16x32xf32>) -> tensor<2x1x16x32xf32>
  return %0 : tensor<2x1x16x32xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op weight tensor must be 1D with size equal to number of channels (32)
func.func @group_norm_weight_wrong_rank(%arg0: tensor<2x1x16x32xf32>, %arg1: tensor<2x32xf32>) -> tensor<2x1x16x32xf32> {
  %0 = "ttnn.group_norm"(%arg0, %arg1) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x1x16x32xf32>, tensor<2x32xf32>) -> tensor<2x1x16x32xf32>
  return %0 : tensor<2x1x16x32xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op weight tensor must be 1D with size equal to number of channels (32)
func.func @group_norm_weight_wrong_size(%arg0: tensor<2x1x16x32xf32>, %arg1: tensor<16xf32>) -> tensor<2x1x16x32xf32> {
  %0 = "ttnn.group_norm"(%arg0, %arg1) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x1x16x32xf32>, tensor<16xf32>) -> tensor<2x1x16x32xf32>
  return %0 : tensor<2x1x16x32xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op bias tensor must be 1D with size equal to number of channels (32)
func.func @group_norm_bias_wrong_rank(%arg0: tensor<2x1x16x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<2x32xf32>) -> tensor<2x1x16x32xf32> {
  %0 = "ttnn.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<2x1x16x32xf32>, tensor<32xf32>, tensor<2x32xf32>) -> tensor<2x1x16x32xf32>
  return %0 : tensor<2x1x16x32xf32>
}

// -----
// CHECK: error: 'ttnn.group_norm' op bias tensor must be 1D with size equal to number of channels (32)
func.func @group_norm_bias_wrong_size(%arg0: tensor<2x1x16x32xf32>, %arg1: tensor<32xf32>, %arg2: tensor<16xf32>) -> tensor<2x1x16x32xf32> {
  %0 = "ttnn.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 8 : i32, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<2x1x16x32xf32>, tensor<32xf32>, tensor<16xf32>) -> tensor<2x1x16x32xf32>
  return %0 : tensor<2x1x16x32xf32>
}
