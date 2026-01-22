// RUN: ttmlir-opt --convert-ttir-to-linalg %s | FileCheck %s

// LayerNorm is decomposed into elementary TOSA ops:
// layer_norm(x) = ((x - mean(x)) / sqrt(var(x) + epsilon)) * weight + bias

module {
  // CHECK-LABEL: func.func @layer_norm_simple
  func.func @layer_norm_simple(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    // Step 1: Compute sum and mean
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reciprocal
    // CHECK: tosa.mul
    // Step 2: Subtract mean (centered)
    // CHECK: tosa.sub
    // Step 3: Compute variance (centered^2, sum, divide)
    // CHECK: tosa.mul
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    // Step 4: Add epsilon and compute rsqrt
    // CHECK: tosa.add
    // CHECK: tosa.rsqrt
    // Step 5: Multiply by inv_std
    // CHECK: tosa.mul
    %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %0 : tensor<512x1024xbf16>
  }

  // CHECK-LABEL: func.func @layer_norm_with_weight
  func.func @layer_norm_with_weight(%arg0: tensor<512x1024xbf16>, %weight: tensor<1024xbf16>) -> tensor<512x1024xbf16> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reciprocal
    // CHECK: tosa.mul
    // CHECK: tosa.sub
    // CHECK: tosa.mul
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    // CHECK: tosa.add
    // CHECK: tosa.rsqrt
    // CHECK: tosa.mul
    // Weight multiplication
    // CHECK: tosa.mul
    %0 = "ttir.layer_norm"(%arg0, %weight) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<512x1024xbf16>, tensor<1024xbf16>) -> tensor<512x1024xbf16>
    return %0 : tensor<512x1024xbf16>
  }

  // CHECK-LABEL: func.func @layer_norm_with_weight_and_bias
  func.func @layer_norm_with_weight_and_bias(%arg0: tensor<512x1024xbf16>, %weight: tensor<1024xbf16>, %bias: tensor<1024xbf16>) -> tensor<512x1024xbf16> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reciprocal
    // CHECK: tosa.mul
    // CHECK: tosa.sub
    // CHECK: tosa.mul
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    // CHECK: tosa.add
    // CHECK: tosa.rsqrt
    // CHECK: tosa.mul
    // Weight multiplication
    // CHECK: tosa.mul
    // Bias addition
    // CHECK: tosa.add
    %0 = "ttir.layer_norm"(%arg0, %weight, %bias) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<512x1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<512x1024xbf16>
    return %0 : tensor<512x1024xbf16>
  }

  // CHECK-LABEL: func.func @layer_norm_3d
  func.func @layer_norm_3d(%arg0: tensor<2x512x1024xbf16>) -> tensor<2x512x1024xbf16> {
    // Normalize over last dimension only
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.rsqrt
    // CHECK: tosa.mul
    %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x512x1024xbf16>) -> tensor<2x512x1024xbf16>
    return %0 : tensor<2x512x1024xbf16>
  }

  // CHECK-LABEL: func.func @layer_norm_multi_dim
  func.func @layer_norm_multi_dim(%arg0: tensor<2x4x8xbf16>) -> tensor<2x4x8xbf16> {
    // Normalize over last two dimensions
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.rsqrt
    // CHECK: tosa.mul
    %0 = "ttir.layer_norm"(%arg0) <{normalized_shape = array<i64: 4, 8>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0>}> : (tensor<2x4x8xbf16>) -> tensor<2x4x8xbf16>
    return %0 : tensor<2x4x8xbf16>
  }
}
