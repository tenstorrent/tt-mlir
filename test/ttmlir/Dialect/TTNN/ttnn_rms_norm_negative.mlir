// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for TTNN RMSNormOp

// CHECK: error: 'ttnn.rms_norm' op input and output must have the same shape
func.func @rms_norm_shape_mismatch(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x16xf32> {
  %1 = "ttnn.rms_norm"(%arg0) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x4x8xf32>) -> tensor<2x4x16xf32>
  return %1 : tensor<2x4x16xf32>
}

// -----
// CHECK: error: 'ttnn.rms_norm' op weight tensor must be 0D for 0D input tensor
func.func @rms_norm_0d_input_weight_not_0d(%arg0: tensor<f32>, %arg1: tensor<1xf32>) -> tensor<f32> {
  %1 = "ttnn.rms_norm"(%arg0, %arg1) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<f32>, tensor<1xf32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----
// CHECK: error: 'ttnn.rms_norm' op bias tensor must be 0D for 0D input tensor
func.func @rms_norm_0d_input_bias_not_0d(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<1xf32>) -> tensor<f32> {
  %1 = "ttnn.rms_norm"(%arg0, %arg1, %arg2) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<f32>, tensor<f32>, tensor<1xf32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// -----
// CHECK: error: 'ttnn.rms_norm' op weight tensor must be 1D with size matching the last dimension of input
func.func @rms_norm_weight_wrong_rank(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8xf32>) -> tensor<2x4x8xf32> {
  %1 = "ttnn.rms_norm"(%arg0, %arg1) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x4x8xf32>, tensor<2x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttnn.rms_norm' op weight tensor must be 1D with size matching the last dimension of input
func.func @rms_norm_weight_wrong_size(%arg0: tensor<2x4x8xf32>, %arg1: tensor<16xf32>) -> tensor<2x4x8xf32> {
  %1 = "ttnn.rms_norm"(%arg0, %arg1) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<2x4x8xf32>, tensor<16xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttnn.rms_norm' op bias tensor must be 1D with size matching the last dimension of input
func.func @rms_norm_bias_wrong_rank(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<2x8xf32>) -> tensor<2x4x8xf32> {
  %1 = "ttnn.rms_norm"(%arg0, %arg1, %arg2) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<2x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttnn.rms_norm' op bias tensor must be 1D with size matching the last dimension of input
func.func @rms_norm_bias_wrong_size(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<4xf32>) -> tensor<2x4x8xf32> {
  %1 = "ttnn.rms_norm"(%arg0, %arg1, %arg2) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<4xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttnn.rms_norm' op weight tensor must be 1D with size matching the last dimension of input
func.func @rms_norm_weight_0d_for_nonzero_input(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<8xf32> {
  %1 = "ttnn.rms_norm"(%arg0, %arg1) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----
// CHECK: error: 'ttnn.rms_norm' op bias tensor must be 1D with size matching the last dimension of input
func.func @rms_norm_bias_0d_for_nonzero_input(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<f32>) -> tensor<8xf32> {
  %1 = "ttnn.rms_norm"(%arg0, %arg1, %arg2) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<8xf32>, tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
