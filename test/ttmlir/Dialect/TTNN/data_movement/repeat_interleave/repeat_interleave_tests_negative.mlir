// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for repeat_interleave operation

// Verify that the parsing fails if the repeats attribute is zero
module {
  func.func @repeat_interleave_repeats_zero(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: error: 'ttnn.repeat_interleave' op Repeats attribute must be non-zero
    %0 = "ttnn.repeat_interleave"(%arg0) {repeats = 0 : ui32, dim = 0 : si32} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// Verify that the parsing fails if operands is a scalar
// -----
module {
  func.func @repeat_interleave_scalar_input(%arg0: tensor<f32>) -> tensor<2xf32>
  {
    // CHECK: error: 'ttnn.repeat_interleave' op Input must be at least a 1D tensor
    %0 = "ttnn.repeat_interleave"(%arg0) {repeats = 2 : ui32, dim = 0 : si32} : (tensor<f32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}

// Verify that the parsing fails if the dim is out of bounds
// -----
module {
  func.func @repeat_interleave_dim_out_of_bounds_1(%arg0: tensor<4xf32>) -> tensor<8xf32>
  {
    // CHECK: error: 'ttnn.repeat_interleave' op Dimension attribute must be within the bounds[-1, 1), got 1
    %0 = "ttnn.repeat_interleave"(%arg0) {repeats = 2 : ui32, dim = 1 : si32} : (tensor<4xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
// -----
module {
  func.func @repeat_interleave_dim_out_of_bounds_2(%arg0: tensor<1x4x1x2xf32>) -> tensor<1x4x1x8xf32>
  {
    // CHECK: error: 'ttnn.repeat_interleave' op Dimension attribute must be within the bounds[-4, 4), got 4
    %0 = "ttnn.repeat_interleave"(%arg0) {repeats = 4 : ui32, dim = 4 : si32} : (tensor<1x4x1x2xf32>) -> tensor<1x4x1x8xf32>
    return %0 : tensor<1x4x1x8xf32>
  }
}

// Verify that the parsing fails if the output shape doesnt match the expected shape
// -----
module {
  func.func @repeat_interleave_output_shape_mismatch_1(%arg0: tensor<4xf32>) -> tensor<4xf32>
  {
    // CHECK: error: 'ttnn.repeat_interleave' op Output shape [4] does not match the expected shape [8]
    %0 = "ttnn.repeat_interleave"(%arg0) {repeats = 2 : ui32, dim = 0 : si32} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
// -----
module {
  func.func @repeat_interleave_output_shape_mismatch_2(%arg0: tensor<1x4x1x2xf32>) -> tensor<1x4x1x2xf32>
  {
    // CHECK: error: 'ttnn.repeat_interleave' op Output shape [1,4,1,2] does not match the expected shape [1,4,4,2]
    %0 = "ttnn.repeat_interleave"(%arg0) {repeats = 4 : ui32, dim = 2 : si32} : (tensor<1x4x1x2xf32>) -> tensor<1x4x1x2xf32>
    return %0 : tensor<1x4x1x2xf32>
  }
}
