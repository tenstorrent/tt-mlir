// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for repeat_interleave operation

// Verify that the parsing fails if the repeats attribute is zero
module {
  func.func @repeat_interleave_repeats_zero(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: error: 'ttnn.repeat_interleave' op Repeats attribute must be non-zero
    %0 = tensor.empty() : tensor<4xf32>
    %1 = "ttnn.repeat_interleave"(%arg0) {repeats = 0 : ui32, dim = 0 : si32} : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// Verify that the parsing fails if operands is a scalar
// -----
module {
  func.func @repeat_interleave_scalar_input(%arg0: tensor<f32>) -> tensor<2xf32>
  {
    // CHECK: error: 'ttnn.repeat_interleave' op Input must be at least a 1D tensor
    %0 = tensor.empty() : tensor<f32>
    %1 = "ttnn.repeat_interleave"(%arg0) {repeats = 2 : ui32, dim = 0 : si32} : (tensor<f32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}

// Verify that the parsing fails if the dim is out of bounds
// -----
module {
  func.func @repeat_interleave_dim_out_of_bounds(%arg0: tensor<4xf32>) -> tensor<8xf32>
  {
    // CHECK: error: 'ttnn.repeat_interleave' op Dimension attribute must be within the bounds(-1, 1), got 1
    %0 = tensor.empty() : tensor<8xf32>
    %1 = "ttnn.repeat_interleave"(%arg0) {repeats = 2 : ui32, dim = 1 : si32} : (tensor<4xf32>) -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}

// Verify that the parsing fails if the output shape doesnt match the expected shape
// -----
module {
  func.func @repeat_interleave_output_shape_mismatch(%arg0: tensor<4xf32>) -> tensor<4xf32>
  {
    // CHECK: error: 'ttnn.repeat_interleave' op Output shape [4] does not match the expected shape [8]
    %0 = tensor.empty() : tensor<4xf32>
    %1 = "ttnn.repeat_interleave"(%arg0) {repeats = 2 : ui32, dim = 0 : si32} : (tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
