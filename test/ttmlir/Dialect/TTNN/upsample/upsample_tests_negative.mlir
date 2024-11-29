// TODO (azecevic): #1385
// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for upsample operation

// Verify that the parsing fails if input or output are not 4D tensors
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_input_3d(%arg0: tensor<3x16x16xbf16>) -> tensor<3x16x16xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected rank of input tensor is 4, got rank 3
    %0 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, 1, 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<3x16x16xbf16>) -> tensor<3x16x16xbf16>
    return %0 : tensor<3x16x16xbf16>
  }
}

// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_input_5d(%arg0: tensor<3x16x16x32x4xbf16>) -> tensor<3x16x16x32x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected rank of input tensor is 4, got rank 5
    %0 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, 1, 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<3x16x16x32x4xbf16>) -> tensor<3x16x16x32x4xbf16>
    return %0 : tensor<3x16x16x32x4xbf16>
  }
}

// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_output_2d(%arg0: tensor<1x16x16x1xbf16>) -> tensor<16x16xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected rank of output tensor is 4, got rank 2
    %0 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, 1, 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<1x16x16x1xbf16>) -> tensor<16x16xbf16>
    return %0 : tensor<16x16xbf16>
  }
}

// Verify that the scale factor is array of 4 ints with first and last element being 1
// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_scale_factor_triplet(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Scale factor must have 4 elements, got 3 elements
    %0 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16>
    return %0 : tensor<3x16x16x4xbf16>
  }
}

// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_scale_factor_dim_n(%arg0: tensor<3x16x16x4xbf16>) -> tensor<6x16x16x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Scale factor must be 1 in the batch (N) dimension, got 2
    %1 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 2, 1, 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<3x16x16x4xbf16>) -> tensor<6x16x16x4xbf16>
    return %1 : tensor<6x16x16x4xbf16>
  }
}

// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_scale_factor_dim_c(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x8xbf16> {
    // CHECK: error: 'ttnn.upsample' op Scale factor must be 1 in the channel (C) dimension, got 2
    %0 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, 1, 1, 2>, operand_constraints = [#any_device, #any_device]}> : (tensor<3x16x16x4xbf16>) -> tensor<3x16x16x8xbf16>
    return %0 : tensor<3x16x16x8xbf16>
  }
}

// Verify that scale factors must be positive integers
// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_nonpositive_scale_factor_uniform(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Scale factors must be positive integers, got (1, -1, 1, 1)
    %0 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, -1, 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<3x16x16x4xbf16>) -> tensor<3x16x16x4xbf16>
    return %0 : tensor<3x16x16x4xbf16>
  }
}

// Verify that output shape must be input shape multiplied by scale factors
// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_shape_mismatch(%arg0: tensor<3x16x16x4xbf16>) -> tensor<3x32x32x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected output shape is (3, 32, 48, 4), got (3, 32, 32, 4)
    %0 = "ttnn.upsample"(%arg0) <{scale_factor = array<i32: 1, 2, 3, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<3x16x16x4xbf16>) -> tensor<3x32x32x4xbf16>
    return %0 : tensor<3x32x32x4xbf16>
  }
}

// Verify that the mode is one of supported modes
// -----
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module {
  func.func @upsample_supported_mode(%arg0: tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16> {
    // CHECK: error: 'ttnn.upsample' op Expected modes are (nearest, bilinear), got "x"
    %0 = "ttnn.upsample"(%arg0) <{mode = "x", scale_factor = array<i32: 1, 1, 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<3x16x32x4xbf16>) -> tensor<3x16x32x4xbf16>
    return %0 : tensor<3x16x32x4xbf16>
  }
}
