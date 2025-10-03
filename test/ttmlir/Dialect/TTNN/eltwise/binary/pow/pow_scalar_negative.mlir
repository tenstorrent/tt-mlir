// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for pow_scalar operation

// Verify that the parsing fails if exponent is negative.

module attributes {} {
  func.func @pow_scalar(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: error: 'ttnn.pow_scalar' op exponent must be non-negative; but got -2
    %1 = "ttnn.pow_scalar"(%arg0) <{rhs = -2 : i32}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}

// -----

module attributes {} {
  func.func @pow_scalar(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: error: 'ttnn.pow_scalar' op exponent must be non-negative; but got -2.0
    %1 = "ttnn.pow_scalar"(%arg0) <{rhs = -2.0 : f32}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
