// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for the ttnn.topk_large_indices verifier.

// Input element type must be bf16.
module {
  func.func @topk_large_indices_bad_input_dtype(%input: tensor<2x64xf32>) -> tensor<2x16xui32> {
    // CHECK: error: 'ttnn.topk_large_indices' op Input element type must be bf16
    %0 = "ttnn.topk_large_indices"(%input) <{k = 16 : ui32}> : (tensor<2x64xf32>) -> tensor<2x16xui32>
    return %0 : tensor<2x16xui32>
  }
}

// -----

// Result element type must be ui32.
module {
  func.func @topk_large_indices_bad_result_dtype(%input: tensor<2x64xbf16>) -> tensor<2x16xi32> {
    // CHECK: error: 'ttnn.topk_large_indices' op Result element type must be ui32
    %0 = "ttnn.topk_large_indices"(%input) <{k = 16 : ui32}> : (tensor<2x64xbf16>) -> tensor<2x16xi32>
    return %0 : tensor<2x16xi32>
  }
}

// -----

// Result last dimension must equal k.
module {
  func.func @topk_large_indices_bad_result_last_dim(%input: tensor<2x64xbf16>) -> tensor<2x32xui32> {
    // CHECK: error: 'ttnn.topk_large_indices' op Result last dimension must equal k
    %0 = "ttnn.topk_large_indices"(%input) <{k = 16 : ui32}> : (tensor<2x64xbf16>) -> tensor<2x32xui32>
    return %0 : tensor<2x32xui32>
  }
}

// -----

// Result leading dimensions must match the input leading dimensions.
module {
  func.func @topk_large_indices_bad_leading_dim(%input: tensor<2x64xbf16>) -> tensor<4x16xui32> {
    // CHECK: error: 'ttnn.topk_large_indices' op Result leading dimensions must match
    %0 = "ttnn.topk_large_indices"(%input) <{k = 16 : ui32}> : (tensor<2x64xbf16>) -> tensor<4x16xui32>
    return %0 : tensor<4x16xui32>
  }
}

// -----

// Input last dimension must be >= k.
module {
  func.func @topk_large_indices_input_smaller_than_k(%input: tensor<2x16xbf16>) -> tensor<2x32xui32> {
    // CHECK: error: 'ttnn.topk_large_indices' op Input last dimension
    %0 = "ttnn.topk_large_indices"(%input) <{k = 32 : ui32}> : (tensor<2x16xbf16>) -> tensor<2x32xui32>
    return %0 : tensor<2x32xui32>
  }
}

// -----

// k must be in [16, 2048] and a multiple of 16 (here k is not a multiple of 16).
module {
  func.func @topk_large_indices_bad_k(%input: tensor<2x64xbf16>) -> tensor<2x20xui32> {
    // CHECK: error: 'ttnn.topk_large_indices' op k must be in [16, 2048] and a multiple of 16
    %0 = "ttnn.topk_large_indices"(%input) <{k = 20 : ui32}> : (tensor<2x64xbf16>) -> tensor<2x20xui32>
    return %0 : tensor<2x20xui32>
  }
}
