// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for grid_sample operation.

// Verify that input must be 4D.
module {
  func.func @grid_sample_input_3d(%arg0: tensor<32x8x8xbf16>, %arg1: tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16> {
    // CHECK: error: 'ttnn.grid_sample' op Input must be a 4D tensor
    %0 = "ttnn.grid_sample"(%arg0, %arg1) <{mode = "bilinear", padding_mode = "zeros", align_corners = false}> : (tensor<32x8x8xbf16>, tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16>
    return %0 : tensor<1x32x6x6xbf16>
  }
}

// -----
// Verify that grid must be 4D.
module {
  func.func @grid_sample_grid_3d(%arg0: tensor<1x32x8x8xbf16>, %arg1: tensor<6x6x2xbf16>) -> tensor<1x32x6x6xbf16> {
    // CHECK: error: 'ttnn.grid_sample' op Grid must be a 4D tensor
    %0 = "ttnn.grid_sample"(%arg0, %arg1) <{mode = "bilinear", padding_mode = "zeros", align_corners = false}> : (tensor<1x32x8x8xbf16>, tensor<6x6x2xbf16>) -> tensor<1x32x6x6xbf16>
    return %0 : tensor<1x32x6x6xbf16>
  }
}

// -----
// Verify that grid's last dimension must be 2.
module {
  func.func @grid_sample_grid_last_dim_3(%arg0: tensor<1x32x8x8xbf16>, %arg1: tensor<1x6x6x3xbf16>) -> tensor<1x32x6x6xbf16> {
    // CHECK: error: 'ttnn.grid_sample' op Grid last dimension must be 2
    %0 = "ttnn.grid_sample"(%arg0, %arg1) <{mode = "bilinear", padding_mode = "zeros", align_corners = false}> : (tensor<1x32x8x8xbf16>, tensor<1x6x6x3xbf16>) -> tensor<1x32x6x6xbf16>
    return %0 : tensor<1x32x6x6xbf16>
  }
}

// -----
// Verify that input and grid must share batch size.
module {
  func.func @grid_sample_batch_mismatch(%arg0: tensor<2x32x8x8xbf16>, %arg1: tensor<3x6x6x2xbf16>) -> tensor<3x32x6x6xbf16> {
    // CHECK: error: 'ttnn.grid_sample' op Input and grid must share the same batch dimension
    %0 = "ttnn.grid_sample"(%arg0, %arg1) <{mode = "bilinear", padding_mode = "zeros", align_corners = false}> : (tensor<2x32x8x8xbf16>, tensor<3x6x6x2xbf16>) -> tensor<3x32x6x6xbf16>
    return %0 : tensor<3x32x6x6xbf16>
  }
}

// -----
// Verify that mode must be one of supported modes.
module {
  func.func @grid_sample_unsupported_mode(%arg0: tensor<1x32x8x8xbf16>, %arg1: tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16> {
    // CHECK: error: 'ttnn.grid_sample' op Expected mode to be one of (bilinear, nearest), got "cubic"
    %0 = "ttnn.grid_sample"(%arg0, %arg1) <{mode = "cubic", padding_mode = "zeros", align_corners = false}> : (tensor<1x32x8x8xbf16>, tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16>
    return %0 : tensor<1x32x6x6xbf16>
  }
}

// -----
// Verify that padding_mode must be one of supported modes.
module {
  func.func @grid_sample_unsupported_padding_mode(%arg0: tensor<1x32x8x8xbf16>, %arg1: tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16> {
    // CHECK: error: 'ttnn.grid_sample' op Expected padding_mode to be one of (zeros, border, reflection), got "wrap"
    %0 = "ttnn.grid_sample"(%arg0, %arg1) <{mode = "bilinear", padding_mode = "wrap", align_corners = false}> : (tensor<1x32x8x8xbf16>, tensor<1x6x6x2xbf16>) -> tensor<1x32x6x6xbf16>
    return %0 : tensor<1x32x6x6xbf16>
  }
}
