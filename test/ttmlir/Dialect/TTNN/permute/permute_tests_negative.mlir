// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for permute operation

// Verfiy that given attribute `permutation` is a valid permutation of the dimensions.
module {
  func.func @permute_non_valid_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttnn.permute' op Expected a permutation of (0, 1, 2), got (0, 1, 0)
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 1, 0>}> : (tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %0 : tensor<16x32x64xbf16>
  }
}

// -----
module {
  func.func @permute_subset_permutation(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttnn.permute' op Expected a permutation of (0, 1, 2), got (0, 1)
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 1>}> : (tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %0 : tensor<16x32x64xbf16>
  }
}

// Verify that the result shape matches the shape of the input tensor after permutation is applied.
// -----
module {
  func.func @permute_non_valid_shape(%arg0: tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16> {
    // CHECK: error: 'ttnn.permute' op Expected result shape (16, 64, 32), got (16, 32, 64)
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 0, 2, 1>}> : (tensor<16x32x64xbf16>) -> tensor<16x32x64xbf16>
    return %0 : tensor<16x32x64xbf16>
  }
}
