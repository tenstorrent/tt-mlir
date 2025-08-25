// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for TTIR RMSNormOp

// CHECK: error: 'ttir.rms_norm' op input and output must have the same shape
func.func @rms_norm_shape_mismatch(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x16xf32> {
  %0 = ttir.empty() : tensor<2x4x16xf32>
  %1 = "ttir.rms_norm"(%arg0, %0) <{normalized_shape = array<i64: 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 1>}> : (tensor<2x4x8xf32>, tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  return %1 : tensor<2x4x16xf32>
}

// -----
// CHECK: error: 'ttir.rms_norm' op normalized_shape cannot be empty
func.func @rms_norm_empty_normalized_shape(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = ttir.empty() : tensor<2x4x8xf32>
  %1 = "ttir.rms_norm"(%arg0, %0) <{normalized_shape = array<i64>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 1>}> : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttir.rms_norm' op normalized_shape has more dimensions than input tensor
func.func @rms_norm_normalized_shape_too_large(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = ttir.empty() : tensor<2x4xf32>
  %1 = "ttir.rms_norm"(%arg0, %0) <{normalized_shape = array<i64: 2, 4, 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 1>}> : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}

// -----
// CHECK: error: 'ttir.rms_norm' op normalized_shape dimensions must match trailing dimensions of input tensor
func.func @rms_norm_normalized_shape_mismatch(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = ttir.empty() : tensor<2x4x8xf32>
  %1 = "ttir.rms_norm"(%arg0, %0) <{normalized_shape = array<i64: 4, 16>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 1>}> : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttir.rms_norm' op normalized_shape dimensions must match trailing dimensions of input tensor
func.func @rms_norm_normalized_shape_single_dim_mismatch(%arg0: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = ttir.empty() : tensor<2x4x8xf32>
  %1 = "ttir.rms_norm"(%arg0, %0) <{normalized_shape = array<i64: 16>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 1>}> : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttir.rms_norm' op weight tensor shape must match normalized_shape
func.func @rms_norm_weight_shape_mismatch(%arg0: tensor<2x4x8xf32>, %arg1: tensor<16xf32>) -> tensor<2x4x8xf32> {
  %0 = ttir.empty() : tensor<2x4x8xf32>
  %1 = "ttir.rms_norm"(%arg0, %arg1, %0) <{normalized_shape = array<i64: 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 1>}> : (tensor<2x4x8xf32>, tensor<16xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttir.rms_norm' op bias tensor shape must match normalized_shape
func.func @rms_norm_bias_shape_mismatch(%arg0: tensor<2x4x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<4xf32>) -> tensor<2x4x8xf32> {
  %0 = ttir.empty() : tensor<2x4x8xf32>
  %1 = "ttir.rms_norm"(%arg0, %arg1, %arg2, %0) <{normalized_shape = array<i64: 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<4xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttir.rms_norm' op weight tensor shape must match normalized_shape
func.func @rms_norm_weight_multidim_shape_mismatch(%arg0: tensor<2x4x8xf32>, %arg1: tensor<4x16xf32>) -> tensor<2x4x8xf32> {
  %0 = ttir.empty() : tensor<2x4x8xf32>
  %1 = "ttir.rms_norm"(%arg0, %arg1, %0) <{normalized_shape = array<i64: 4, 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 1>}> : (tensor<2x4x8xf32>, tensor<4x16xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}

// -----
// CHECK: error: 'ttir.rms_norm' op bias tensor shape must match normalized_shape
func.func @rms_norm_bias_multidim_shape_mismatch(%arg0: tensor<2x4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<2x8xf32>) -> tensor<2x4x8xf32> {
  %0 = ttir.empty() : tensor<2x4x8xf32>
  %1 = "ttir.rms_norm"(%arg0, %arg1, %arg2, %0) <{normalized_shape = array<i64: 4, 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<2x4x8xf32>, tensor<4x8xf32>, tensor<2x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  return %1 : tensor<2x4x8xf32>
}
