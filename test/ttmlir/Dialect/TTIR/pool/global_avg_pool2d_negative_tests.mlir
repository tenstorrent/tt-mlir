// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for global_avg_pool2d operation

// Test 1: GlobalAvgPool2dOp with invalid tensor rank (input has rank 1)
module {
  func.func @global_avg_pool2d_invalid_input_rank_1d(%arg0: tensor<64xbf16>) -> tensor<1x1x1x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x64xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op input tensor must have at least 2 dimensions for global average pooling over 2 spatial dimensions
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
    return %1 : tensor<1x1x1x64xbf16>
  }
}

// -----
// Test 2: GlobalAvgPool2dOp with batch dimension mismatch
module {
  func.func @global_avg_pool2d_batch_mismatch(%arg0: tensor<2x32x32x64xbf16>) -> tensor<1x1x1x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x64xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op batch dimension must remain the same between input and output
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<2x32x32x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
    return %1 : tensor<1x1x1x64xbf16>
  }
}

// -----
// Test 3: GlobalAvgPool2dOp with channel dimension mismatch
module {
  func.func @global_avg_pool2d_channel_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x1x1x128xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x128xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op channel dimension must remain the same between input and output
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x32x32x64xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    return %1 : tensor<1x1x1x128xbf16>
  }
}

// -----
// Test 4: GlobalAvgPool2dOp with both batch and channel dimension mismatch
module {
  func.func @global_avg_pool2d_batch_channel_mismatch(%arg0: tensor<2x32x32x64xbf16>) -> tensor<4x1x1x128xbf16> {
    %0 = ttir.empty() : tensor<4x1x1x128xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op batch dimension must remain the same between input and output
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<2x32x32x64xbf16>, tensor<4x1x1x128xbf16>) -> tensor<4x1x1x128xbf16>
    return %1 : tensor<4x1x1x128xbf16>
  }
}

// -----
// Test 5: GlobalAvgPool2dOp with output spatial dimensions not reduced to 1x1
module {
  func.func @global_avg_pool2d_output_not_1x1(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x2x2x64xbf16> {
    %0 = ttir.empty() : tensor<1x2x2x64xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x32x32x64xbf16>, tensor<1x2x2x64xbf16>) -> tensor<1x2x2x64xbf16>
    return %1 : tensor<1x2x2x64xbf16>
  }
}

// -----
// Test 6: GlobalAvgPool2dOp with output height not reduced to 1
module {
  func.func @global_avg_pool2d_output_height_not_1(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x32x1x64xbf16> {
    %0 = ttir.empty() : tensor<1x32x1x64xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x32x32x64xbf16>, tensor<1x32x1x64xbf16>) -> tensor<1x32x1x64xbf16>
    return %1 : tensor<1x32x1x64xbf16>
  }
}

// -----
// Test 7: GlobalAvgPool2dOp with output width not reduced to 1
module {
  func.func @global_avg_pool2d_output_width_not_1(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x32x64xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x32x32x64xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x1x32x64xbf16>
    return %1 : tensor<1x1x32x64xbf16>
  }
}

// -----
// Test 8: GlobalAvgPool2dOp with 3D input (batch, spatial, channels) - batch mismatch
module {
  func.func @global_avg_pool2d_3d_batch_mismatch(%arg0: tensor<2x256x64xbf16>) -> tensor<1x1x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x64xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op batch dimension must remain the same between input and output
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<2x256x64xbf16>, tensor<1x1x64xbf16>) -> tensor<1x1x64xbf16>
    return %1 : tensor<1x1x64xbf16>
  }
}

// -----
// Test 9: GlobalAvgPool2dOp with 3D input - channel mismatch
module {
  func.func @global_avg_pool2d_3d_channel_mismatch(%arg0: tensor<1x256x64xbf16>) -> tensor<1x1x128xbf16> {
    %0 = ttir.empty() : tensor<1x1x128xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op channel dimension must remain the same between input and output
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x256x64xbf16>, tensor<1x1x128xbf16>) -> tensor<1x1x128xbf16>
    return %1 : tensor<1x1x128xbf16>
  }
}

// -----
// Test 10: GlobalAvgPool2dOp with 5D input - batch mismatch
module {
  func.func @global_avg_pool2d_5d_batch_mismatch(%arg0: tensor<2x8x8x8x64xbf16>) -> tensor<1x1x1x1x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x1x64xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op batch dimension must remain the same between input and output
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<2x8x8x8x64xbf16>, tensor<1x1x1x1x64xbf16>) -> tensor<1x1x1x1x64xbf16>
    return %1 : tensor<1x1x1x1x64xbf16>
  }
}

// -----
// Test 11: GlobalAvgPool2dOp with 5D input - channel mismatch
module {
  func.func @global_avg_pool2d_5d_channel_mismatch(%arg0: tensor<1x8x8x8x64xbf16>) -> tensor<1x1x1x1x32xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x1x32xbf16>
    // CHECK: error: 'ttir.global_avg_pool2d' op channel dimension must remain the same between input and output
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x8x8x8x64xbf16>, tensor<1x1x1x1x32xbf16>) -> tensor<1x1x1x1x32xbf16>
    return %1 : tensor<1x1x1x1x32xbf16>
  }
}
