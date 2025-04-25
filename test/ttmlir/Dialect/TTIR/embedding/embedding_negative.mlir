// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Verify that input tensor is at most 2D tensor.
module {
  func.func @embedding_input_3D(%input: tensor<2x4x8xui32>, %weight: tensor<5x10xbf16>) -> tensor<4x8x10xbf16> {
    %output = ttir.empty() : tensor<4x8x10xbf16>
    // CHECK: error: 'ttir.embedding' op input must be at most a 2D tensor, got 3D ranked tensor
    %result = "ttir.embedding"(%input, %weight, %output) : (tensor<2x4x8xui32>, tensor<5x10xbf16>, tensor<4x8x10xbf16>) -> tensor<4x8x10xbf16>
    return %result : tensor<4x8x10xbf16>
  }
}

// -----
// Verify that the weight tensor is at least 2D tensor.
module {
  func.func @embedding_weight_1D(%input: tensor<2x4xui32>, %weight: tensor<10xbf16>) -> tensor<2x4x10xbf16> {
    %output = ttir.empty() : tensor<2x4x10xbf16>
    // CHECK: error: 'ttir.embedding' op weight must be at least 2D tensor, got 1D ranked tensor
    %result = "ttir.embedding"(%input, %weight, %output) : (tensor<2x4xui32>, tensor<10xbf16>, tensor<2x4x10xbf16>) -> tensor<2x4x10xbf16>
    return %result : tensor<2x4x10xbf16>
  }
}

// -----
// Verify that all dims of the weight tensor except the last two are equal to 1.
module {
  func.func @embedding_weight_non_2D_effective(%input: tensor<2x4xui32>, %weight: tensor<1x2x3x4xbf16>) -> tensor<2x4x4xbf16> {
    %output = ttir.empty() : tensor<2x4x4xbf16>
    // CHECK: error: 'ttir.embedding' op weight must be effectively 2D tensor
    %result = "ttir.embedding"(%input, %weight, %output) : (tensor<2x4xui32>, tensor<1x2x3x4xbf16>, tensor<2x4x4xbf16>) -> tensor<2x4x4xbf16>
    return %result : tensor<2x4x4xbf16>
  }
}

// -----
// Verify that the output shape is equal to the input shape with the addition of the last dim of the weight tensor.
module {
  func.func @embedding_output_shape(%input: tensor<2x4xui32>, %weight: tensor<1x5x10xbf16>) -> tensor<2x4x5x10xbf16> {
    %output = ttir.empty() : tensor<2x4x5x10xbf16>
    // CHECK: error: 'ttir.embedding' op expected output shape of (2, 4, 10) but got (2, 4, 5, 10)
    %result = "ttir.embedding"(%input, %weight, %output) : (tensor<2x4xui32>, tensor<1x5x10xbf16>, tensor<2x4x5x10xbf16>) -> tensor<2x4x5x10xbf16>
    return %result : tensor<2x4x5x10xbf16>
  }
}
