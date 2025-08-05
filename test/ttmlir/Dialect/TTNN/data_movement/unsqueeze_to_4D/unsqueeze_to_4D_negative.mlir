// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Test negative cases for ttnn.unsqueeze_to_4D operation.

module {
  func.func @output_not_4d(%arg0: tensor<32x128xbf16>) -> tensor<1x32x128xbf16> {
    // CHECK: error: 'ttnn.unsqueeze_to_4D' op Output tensor must be 4D, but has rank 3
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<32x128xbf16>) -> tensor<1x32x128xbf16>
    return %0 : tensor<1x32x128xbf16>
  }
}

// -----

module {
  func.func @input_rank_too_high(%arg0: tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4xbf16> {
    // CHECK: error: 'ttnn.unsqueeze_to_4D' op Input tensor rank 5 is greater than 4
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4xbf16>
    return %0 : tensor<1x2x3x4xbf16>
  }
}

// -----

module {
  func.func @dimension_mismatch(%arg0: tensor<32x128xbf16>) -> tensor<1x1x64x64xbf16> {
    // CHECK: error: 'ttnn.unsqueeze_to_4D' op Input dimension 0 with size 32 does not match output dimension 2 with size 64
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<32x128xbf16>) -> tensor<1x1x64x64xbf16>
    return %0 : tensor<1x1x64x64xbf16>
  }
}

// -----

module {
  func.func @leading_dim_not_one(%arg0: tensor<32x128xbf16>) -> tensor<2x1x32x128xbf16> {
    // CHECK: error: 'ttnn.unsqueeze_to_4D' op Leading dimension 0 must be 1, but is 2
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<32x128xbf16>) -> tensor<2x1x32x128xbf16>
    return %0 : tensor<2x1x32x128xbf16>
  }
}

// -----

module {
  func.func @element_count_mismatch(%arg0: tensor<32x128xbf16>) -> tensor<1x1x32x256xbf16> {
    // CHECK: error: 'ttnn.unsqueeze_to_4D' op Input dimension 1 with size 128 does not match output dimension 3 with size 256
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<32x128xbf16>) -> tensor<1x1x32x256xbf16>
    return %0 : tensor<1x1x32x256xbf16>
  }
}