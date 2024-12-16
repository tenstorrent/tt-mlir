// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for slice operation

// Verify that the parsing fails if the begins attribute is not a 3D tensor
module attributes {} {
  func.func @slice_negative_invalid_shape(%arg0: tensor<bf16>) -> tensor<1xbf16> {
    %0 = tensor.empty() : tensor<1xbf16>
    // CHECK: error: 'ttir.slice' op Input must be at least a 1D tensor
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32], ends = [0: i32], step = [1: i32]}> : (tensor<bf16>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }
}

// Verify that the parsing fails if the begins size is not equal to the input tensor rank
// -----
module attributes {} {
  func.func @slice_negative_invalid_begins(%arg0: tensor<3x128x64xbf16>) -> tensor<1x64x64xbf16> {
    %0 = tensor.empty() : tensor<1x64x64xbf16>
    // CHECK: error: 'ttir.slice' op Begins, ends, and step attributes must have the same number of elements as the input tensor rank
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32], ends = [0: i32, 63: i32, 63: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<3x128x64xbf16>, tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
    return %1 : tensor<1x64x64xbf16>
  }
}

// Verify that the parsing fails if the ends size is not equal to the input tensor rank
// -----
module attributes {} {
  func.func @slice_negative_invalid_ends(%arg0: tensor<3x128x64xbf16>) -> tensor<1x64x64xbf16> {
    %0 = tensor.empty() : tensor<1x64x64xbf16>
    // CHECK: error: 'ttir.slice' op Begins, ends, and step attributes must have the same number of elements as the input tensor rank
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [0: i32, 63: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<3x128x64xbf16>, tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
    return %1 : tensor<1x64x64xbf16>
  }
}

// Verify that the parsing fails if the step size is not equal to the input tensor rank
// -----
module attributes {} {
  func.func @slice_negative_invalid_step(%arg0: tensor<3x128x64xbf16>) -> tensor<1x64x64xbf16> {
    %0 = tensor.empty() : tensor<1x64x64xbf16>
    // CHECK: error: 'ttir.slice' op Begins, ends, and step attributes must have the same number of elements as the input tensor rank
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [0: i32, 63: i32, 63: i32], step = [1: i32, 1: i32]}> : (tensor<3x128x64xbf16>, tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
    return %1 : tensor<1x64x64xbf16>
  }
}

// Verify that the parsing fails if the output type is not equal to the input tensor type
// -----
module attributes {} {
  func.func @slice_negative_invalid_output_datatype(%arg0: tensor<3x128x64xbf16>) -> tensor<1x64x64xf32> {
    %0 = tensor.empty() : tensor<1x64x64xf32>
    // CHECK: error: 'ttir.slice' op Output tensor must have the same element type as the input tensor
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [0: i32, 63: i32, 63: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<3x128x64xbf16>, tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
    return %1 : tensor<1x64x64xf32>
  }
}

// Verify that the parsing fails if the output rank is not equal to the input tensor rank
// -----
module attributes {} {
  func.func @slice_negative_input_output_rank_missmatch(%arg0: tensor<3x128x64xbf16>) -> tensor<1x1x64x64xbf16> {
    %0 = tensor.empty() : tensor<1x1x64x64xbf16>
    // CHECK: error: 'ttir.slice' op Output tensor must have the same rank as the input tensor
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [0: i32, 63: i32, 63: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<3x128x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<1x1x64x64xbf16>
    return %1 : tensor<1x1x64x64xbf16>
  }
}

// Verify that the parsing fails if the begin value exceeds positive limit
// -----
module attributes {} {
  func.func @slice_negative_invalid_begin_positive(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x8xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x8xbf16>
    // CHECK: error: 'ttir.slice' op Invalid begin index for dimension 2. Expected value in range [-128, 128), got 128. Input shape: (10, 3, 128, 64)
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 128: i32, 32: i32], ends = [10: i32, 3: i32, 128: i32, 64: i32], step = [3: i32, 3: i32, 8: i32, 4: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x8xbf16>) -> tensor<4x1x16x8xbf16>
    return %1 : tensor<4x1x16x8xbf16>
  }
}

// Verify that the parsing fails if the begin value exceeds negative limit
// -----
module attributes {} {
  func.func @slice_negative_invalid_begin_negative(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x8xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x8xbf16>
    // CHECK: error: 'ttir.slice' op Invalid begin index for dimension 2. Expected value in range [-128, 128), got -129. Input shape: (10, 3, 128, 64)
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, -129: i32, 32: i32], ends = [10: i32, 3: i32, 128: i32, 64: i32], step = [3: i32, 3: i32, 8: i32, 4: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x8xbf16>) -> tensor<4x1x16x8xbf16>
    return %1 : tensor<4x1x16x8xbf16>
  }
}

// Verify that the parsing fails if the end value exceeds positive limit
// -----
module attributes {} {
  func.func @slice_negative_invalid_end_positive(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x8xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x8xbf16>
    // CHECK: error: 'ttir.slice' op Invalid end index for dimension 3. Expected value in range [-64, 64], got 65. Input shape: (10, 3, 128, 64)
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32, 32: i32], ends = [10: i32, 3: i32, 128: i32, 65: i32], step = [3: i32, 3: i32, 8: i32, 4: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x8xbf16>) -> tensor<4x1x16x8xbf16>
    return %1 : tensor<4x1x16x8xbf16>
  }
}

// Verify that the parsing fails if the end value exceeds negative limit
// -----
module attributes {} {
  func.func @slice_negative_invalid_end_negative(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x8xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x8xbf16>
    // CHECK: error: 'ttir.slice' op Invalid end index for dimension 3. Expected value in range [-64, 64], got -65. Input shape: (10, 3, 128, 64)
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32, 32: i32], ends = [10: i32, 3: i32, 128: i32, -65: i32], step = [3: i32, 3: i32, 8: i32, 4: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x8xbf16>) -> tensor<4x1x16x8xbf16>
    return %1 : tensor<4x1x16x8xbf16>
  }
}

// Verify that the parsing fails if the step value is equal to zero
// -----
module attributes {} {
  func.func @slice_negative_step_is_zero(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x8xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x8xbf16>
    // CHECK: error: 'ttir.slice' op Step value for dimension 3 cannot be zero
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32, 32: i32], ends = [10: i32, 3: i32, 128: i32, 64: i32], step = [3: i32, 3: i32, 8: i32, 0: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x8xbf16>) -> tensor<4x1x16x8xbf16>
    return %1 : tensor<4x1x16x8xbf16>
  }
}

// Verify that the parsing fails if the begin index is greater than end and step is positive
// -----
module attributes {} {
  func.func @slice_negative_begin_greater_than_end_positive_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x8xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x8xbf16>
    // CHECK: error: 'ttir.slice' op For positive step, begin index must be less than or equal to end index for dimension 0. Got begin: 9, end: 0, step: 3, input shape: (10, 3, 128, 64)
    %1 = "ttir.slice"(%arg0, %0) <{begins = [9: i32, 0: i32, 0: i32, 32: i32], ends = [0: i32, 3: i32, 32: i32, 64: i32], step = [3: i32, 3: i32, 8: i32, 4: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x8xbf16>) -> tensor<4x1x16x8xbf16>
    return %1 : tensor<4x1x16x8xbf16>
  }
}

// -----
module attributes {} {
  func.func @slice_negative_begin_greater_than_end_positive_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x8x8xbf16> {
    %0 = tensor.empty() : tensor<4x1x8x8xbf16>
    // CHECK: error: 'ttir.slice' op For positive step, begin index must be less than or equal to end index for dimension 2. Got begin: 96 (-32), end: 32 (-96), step: 8, input shape: (10, 3, 128, 64)
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, -32: i32, 32: i32], ends = [10: i32, 3: i32, -96: i32, 64: i32], step = [3: i32, 3: i32, 8: i32, 4: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x8x8xbf16>) -> tensor<4x1x8x8xbf16>
    return %1 : tensor<4x1x8x8xbf16>
  }
}

// Verify that the parsing fails if the end index is greater than begin and step is negative
// -----
module attributes {} {
  func.func @slice_negative_begin_less_than_end_negative_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x8xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x8xbf16>
    // CHECK: error: 'ttir.slice' op For negative step, begin index must be greater than or equal to end index for dimension 1. Got begin: 0 (-3), end: 2 (-1), step: -3, input shape: (10, 3, 128, 64)
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, -3: i32, 0: i32, 32: i32], ends = [10: i32, -1: i32, 32: i32, 128: i32], step = [3: i32, -3: i32, 8: i32, 8: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x8xbf16>) -> tensor<4x1x16x8xbf16>
    return %1 : tensor<4x1x16x8xbf16>
  }
}

// -----
module attributes {} {
  func.func @slice_negative_begin_less_than_end_negative_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<5x1x16x8xbf16> {
    %0 = tensor.empty() : tensor<5x1x16x8xbf16>
    // CHECK: error: 'ttir.slice' op For negative step, begin index must be greater than or equal to end index for dimension 0. Got begin: 0, end: 10, step: -2, input shape: (10, 3, 128, 64)
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32, 32: i32], ends = [10: i32, 3: i32, 128: i32, 64: i32], step = [-2: i32, 3: i32, 8: i32, 4: i32]}> : (tensor<10x3x128x64xbf16>, tensor<5x1x16x8xbf16>) -> tensor<5x1x16x8xbf16>
    return %1 : tensor<5x1x16x8xbf16>
  }
}

// Verify that the parsing fails if there is missmatch in output dimension
// -----
module attributes {} {
  func.func @slice_negative_invalid_output_shape(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x16xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x16xbf16>
    // CHECK: error: 'ttir.slice' op Mismatch in dimension 3 of the output tensor: expected size 8, but got 16
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32, 32: i32], ends = [10: i32, 3: i32, 128: i32, 64: i32], step = [3: i32, 3: i32, 8: i32, 4: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x16xbf16>) -> tensor<4x1x16x16xbf16>
    return %1 : tensor<4x1x16x16xbf16>
  }
}
