// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for upsample operation

// Verify that the parsing fails if input or output are not 4D tensors.
module {
  func.func @upsample_input_3d(%arg0: tensor<3x16x16xbf16>) -> tensor<3x16x16xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected rank of input tensor is 4, got rank 3
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = 1 : si32}> : (tensor<3x16x16xbf16>) -> tensor<3x16x16xbf16>
    return %1 : tensor<3x16x16xbf16>
  }
}

// -----
module {
  func.func @upsample_input_5d(%arg0: tensor<3x16x16x32x4xbf16>) -> tensor<3x16x16x32x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected rank of input tensor is 4, got rank 5
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = 1 : si32}> : (tensor<3x16x16x32x4xbf16>) -> tensor<3x16x16x32x4xbf16>
    return %1 : tensor<3x16x16x32x4xbf16>
  }
}

// -----
module {
  func.func @upsample_output_2d(%arg0: tensor<1x16x16x1xbf16>) -> tensor<16x16xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected rank of output tensor is 4, got rank 2
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = 1 : si32}> : (tensor<1x16x16x1xbf16>) -> tensor<16x16xbf16>
    return %1 : tensor<16x16xbf16>
  }
}

// Verify that the scale factor is either integer or pair of integers.
// -----
module {
  func.func @upsample_scale_factor_triplet(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected integer or pair of integers, got tuple of size 3
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, 1, 1>}> : (tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16>
    return %1 : tensor<3x16x16x4xbf16>
  }
}

// Verify that scale factors must be positive integers.
// -----
module {
  func.func @upsample_nonpositive_scale_factor_uniform(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Scale factors H = 0 and W = 0 must be positive integers
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = 0 : si32}> : (tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16>
    return %1 : tensor<3x16x16x4xbf16>
  }
}

// -----
module {
  func.func @upsample_nonpositive_scale_factor_nonuniform(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Scale factors H = 1 and W = -1 must be positive integers
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, -1>}> : (tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16>
    return %1 : tensor<3x16x16x4xbf16>
  }
}

// Verify that output shape must be input shape multiplied by scale factors.
// -----
module {
  func.func @upsample_mismatch_n(%arg0: tensor<3x16x16x4xbf16>) -> tensor<6x16x16x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected output N dimension to be 3, got 6
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = 1 : si32}> : (tensor<3x16x16x4xbf16>) -> tensor<6x16x16x4xbf16>
    return %1 : tensor<6x16x16x4xbf16>
  }
}

// -----
module {
  func.func @upsample_mismatch_c(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x8xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected output C dimension to be 4, got 8
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = 1 : si32}> : (tensor<3x16x16x4xbf16>) -> tensor<3x16x16x8xbf16>
    return %1 : tensor<3x16x16x8xbf16>
  }
}

// -----
module {
  func.func @upsample_unifrom_scale_mismatch_h(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected output H dimension to be input H dimension * scaleH = 32, got 16
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = 2 : si32}> : (tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16>
    return %1 : tensor<3x16x32x4xbf16>
  }
}

// -----
module {
  func.func @upsample_unifrom_scale_mismatch_w(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x32x32x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected output W dimension to be input W dimension * scaleW = 64, got 32
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = 2 : si32}> : (tensor<3x16x32x4xbf16>) -> tensor<3x32x32x4xbf16>
    return %1 : tensor<3x32x32x4xbf16>
  }
}

// -----
module {
  func.func @upsample_nonunifrom_scale_mismatch_h(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x16x128x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected output H dimension to be input H dimension * scaleH = 32, got 16
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32 : 2, 4>}> : (tensor<3x16x32x4xbf16>) -> tensor<3x16x128x4xbf16>
    return %1 : tensor<3x16x128x4xbf16>
  }
}

// -----
module {
  func.func @upsample_nonunifrom_scale_mismatch_w(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x32x64x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected output W dimension to be input W dimension * scaleW = 128, got 64
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32 : 2, 4>}> : (tensor<3x16x32x4xbf16>) -> tensor<3x32x64x4xbf16>
    return %1 : tensor<3x32x64x4xbf16>
  }
}

// Verify that the mode is one of supported modes.
// -----
module {
  func.func @upsample_supported_mode(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected modes are (nearest, bilinear), got "x"
    %0 = tensor.empty() : tensor<3x16x32x4xbf16>
    %1 = "ttnn.upsample"(%arg0) <{mode = "x", scale_factor = 1 : si32}> : (tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16>
    return %1 : tensor<3x16x32x4xbf16>
  }
}
