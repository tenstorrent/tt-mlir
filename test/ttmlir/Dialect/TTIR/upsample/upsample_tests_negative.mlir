// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for upsample operation

// Verify that the parsing fails if input or output are not 4D tensors
module {
  func.func @upsample_input_3d(%arg0: tensor<3x16x16xbf16>) -> tensor<3x16x16xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected rank of input tensor is 4, got rank 3
    %0 = tensor.empty() : tensor<3x16x16xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x16xbf16>, tensor<3x16x16xbf16>) -> tensor<3x16x16xbf16>
    return %1 : tensor<3x16x16xbf16>
  }
}

// -----
module {
  func.func @upsample_input_5d(%arg0: tensor<3x16x16x32x4xbf16>) -> tensor<3x16x16x32x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected rank of input tensor is 4, got rank 5
    %0 = tensor.empty() : tensor<3x16x16x32x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x16x32x4xbf16>, tensor<3x16x16x32x4xbf16>) -> tensor<3x16x16x32x4xbf16>
    return %1 : tensor<3x16x16x32x4xbf16>
  }
}

// -----
module {
  func.func @upsample_output_2d(%arg0: tensor<1x16x16x1xbf16>) -> tensor<16x16xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected rank of output tensor is 4, got rank 2
    %0 = tensor.empty() : tensor<16x16xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<1x16x16x1xbf16>, tensor<16x16xbf16>) -> tensor<16x16xbf16>
    return %1 : tensor<16x16xbf16>
  }
}

// Verify that the scale factor is either int or pair of ints
// -----
module {
  func.func @upsample_scale_factor_triplet(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected int or pair of ints, got tuple of size 3
    %0 = tensor.empty() : tensor<3x16x16x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32: 1, 1, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x16x4xbf16>, tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16>
    return %1 : tensor<3x16x16x4xbf16>
  }
}

// Verify that scale factors must be positive integers
// -----
module {
  func.func @upsample_nonpositive_scale_factor_uniform(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Scale factors H = 0 and W = 0 must be positive integers
    %0 = tensor.empty() : tensor<3x16x16x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 0 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x16x4xbf16>, tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16>
    return %1 : tensor<3x16x16x4xbf16>
  }
}

// -----
module {
  func.func @upsample_nonpositive_scale_factor_nonuniform(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Scale factors H = 1 and W = -1 must be positive integers
    %0 = tensor.empty() : tensor<3x16x16x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32: 1, -1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x16x4xbf16>, tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16>
    return %1 : tensor<3x16x16x4xbf16>
  }
}

// Verify that output shape must be input shape multiplied by scale factors
// -----
module {
  func.func @upsample_mismatch_n(%arg0: tensor<3x16x16x4xbf16>) -> tensor<6x16x16x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected output N dimension to be 3, got 6
    %0 = tensor.empty() : tensor<6x16x16x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x16x4xbf16>, tensor<6x16x16x4xbf16>) -> tensor<6x16x16x4xbf16>
    return %1 : tensor<6x16x16x4xbf16>
  }
}

// -----
module {
  func.func @upsample_mismatch_c(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x8xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected output C dimension to be 4, got 8
    %0 = tensor.empty() : tensor<3x16x16x8xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x16x4xbf16>, tensor<3x16x16x8xbf16>) -> tensor<3x16x16x8xbf16>
    return %1 : tensor<3x16x16x8xbf16>
  }
}

// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample__mismatch_c_channel_first(%arg0: tensor<3x4x16x16xbf16>) -> tensor<3x8x16x16xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected output C dimension to be 4, got 8
    %0 = tensor.empty() : tensor<3x8x16x16xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 1 : si32, channel_last = false, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x4x16x16xbf16>, tensor<3x8x16x16xbf16>) -> tensor<3x8x16x16xbf16>
    return %1 : tensor<3x8x16x16xbf16>
  }
}

// -----
module {
  func.func @upsample_unifrom_scale_mismatch_h(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected output H dimension to be input H dimension * scaleH = 32, got 16
    %0 = tensor.empty() : tensor<3x16x32x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 2 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x32x4xbf16>, tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16>
    return %1 : tensor<3x16x32x4xbf16>
  }
}

// -----
module {
  func.func @upsample_unifrom_scale_mismatch_w(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x32x32x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected output W dimension to be input W dimension * scaleW = 64, got 32
    %0 = tensor.empty() : tensor<3x32x32x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 2 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x32x4xbf16>, tensor<3x32x32x4xbf16>) -> tensor<3x32x32x4xbf16>
    return %1 : tensor<3x32x32x4xbf16>
  }
}

// -----
module {
  func.func @upsample_nonunifrom_scale_mismatch_h(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x16x128x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected output H dimension to be input H dimension * scaleH = 32, got 16
    %0 = tensor.empty() : tensor<3x16x128x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32 : 2, 4>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x32x4xbf16>, tensor<3x16x128x4xbf16>) -> tensor<3x16x128x4xbf16>
    return %1 : tensor<3x16x128x4xbf16>
  }
}

// -----
module {
  func.func @upsample_nonunifrom_scale_mismatch_w(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x32x64x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected output W dimension to be input W dimension * scaleW = 128, got 64
    %0 = tensor.empty() : tensor<3x32x64x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32 : 2, 4>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x32x4xbf16>, tensor<3x32x64x4xbf16>) -> tensor<3x32x64x4xbf16>
    return %1 : tensor<3x32x64x4xbf16>
  }
}

// Verify that the mode is one of supported modes
// -----
module {
  func.func @upsample_supported_mode(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16> {
    // CHECK: error: 'ttir.upsample' op Expected modes are (nearest, bilinear), got "x"
    %0 = tensor.empty() : tensor<3x16x32x4xbf16>
    %1 = "ttir.upsample"(%arg0, %0) <{mode = "x", scale_factor = 1 : si32, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<3x16x32x4xbf16>, tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16>
    return %1 : tensor<3x16x32x4xbf16>
  }
}
