// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for rotary_embedding operation.

// Verify that the parsing fails if cos and sin have different data types.
module {
  func.func @dtype_mismatch(%input: tensor<1x1x32x32xbf16>, %cos: tensor<1x1x32x32xbf16>, %sin: tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.rotary_embedding' op cos and sin tensor dtypes must match.
    %0 = "ttnn.rotary_embedding"(%input, %cos, %sin) : (tensor<1x1x32x32xbf16>, tensor<1x1x32x32xbf16>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}

// -----

// Verify that the parsing fails if cos and sin have different shapes.
module {
  func.func @shape_mismatch(%input: tensor<1x1x32x32xbf16>, %cos: tensor<1x1x32x64xbf16>, %sin: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16> {
    // CHECK: error: 'ttnn.rotary_embedding' op cos and sin tensor shapes must match.
    %0 = "ttnn.rotary_embedding"(%input, %cos, %sin) : (tensor<1x1x32x32xbf16>, tensor<1x1x32x64xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16>
    return %0 : tensor<1x1x32x32xbf16>
  }
}
