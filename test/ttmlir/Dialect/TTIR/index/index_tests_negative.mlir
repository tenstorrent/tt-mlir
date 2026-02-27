// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for index operation

// Verify that the parsing fails if the begins attribute is not a 3D tensor
module attributes {} {
  func.func @index_negative_invalid_shape(%arg0: tensor<bf16>) -> tensor<1xbf16> {
    // CHECK: error: 'ttir.index' op Input must be at least a 1D tensor
    %1 = "ttir.index"(%arg0) <{dim = 0: i32, begin = 0: i32, end = 0: i32, step = 1: i32}> : (tensor<bf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }
}

// Verify that the parsing fails if the dim is not in the rank range of the input tensor
// -----
module attributes {} {
  func.func @index_negative_invalid_begins(%arg0: tensor<3x128x64xbf16>) -> tensor<3x128x64xbf16> {
    // CHECK: error: 'ttir.index' op Invalid dimension index 3. Input tensor rank is 3
    %1 = "ttir.index"(%arg0) <{dim = 3 : i32, begin = 0: i32, end = 0: i32, step = 1: i32}> : (tensor<3x128x64xbf16>) -> tensor<3x128x64xbf16>
    return %1 : tensor<3x128x64xbf16>
  }
}

// Verify that the parsing fails if the output type is not equal to the input tensor type
// -----
module attributes {} {
  func.func @index_negative_invalid_output_datatype(%arg0: tensor<3x128x64xbf16>) -> tensor<3x128x32xf32> {
    // CHECK: error: 'ttir.index' op Output tensor must have the same element type as the input tensor
    %1 = "ttir.index"(%arg0) <{dim = 2 : i32, begin = 0: i32, end = 32: i32, step = 1: i32}> : (tensor<3x128x64xbf16>) -> tensor<3x128x32xf32>
    return %1 : tensor<3x128x32xf32>
  }
}

// Verify that the parsing fails if the output rank is not equal to the input tensor rank
// -----
module attributes {} {
  func.func @index_negative_input_output_rank_missmatch(%arg0: tensor<3x128x64xbf16>) -> tensor<3x64x64x1xbf16> {
    // CHECK: error: 'ttir.index' op Output tensor must have the same rank as the input tensor
    %1 = "ttir.index"(%arg0) <{dim = 1: i32, begin = 0: i32, end = 64: i32, step = 1: i32}> : (tensor<3x128x64xbf16>) -> tensor<3x64x64x1xbf16>
    return %1 : tensor<3x64x64x1xbf16>
  }
}

// Verify that the parsing fails if the begin value exceeds positive limit
// -----
module attributes {} {
  func.func @index_negative_invalid_begin_positive(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x1x128x64xbf16> {
    // CHECK: error: 'ttir.index' op Invalid begin index for dimension 1. Expected value in range [-3, 3), got 3. Input shape: (10, 3, 128, 64)
    %1 = "ttir.index"(%arg0) <{dim = 1: i32, begin = 3: i32, end = 3: i32, step = 1: i32}> : (tensor<10x3x128x64xbf16>) -> tensor<10x1x128x64xbf16>
    return %1 : tensor<10x1x128x64xbf16>
  }
}

// Verify that the parsing fails if the begin value exceeds negative limit
// -----
module attributes {} {
  func.func @index_negative_invalid_begin_negative(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x64x64xbf16> {
    // CHECK: error: 'ttir.index' op Invalid begin index for dimension 2. Expected value in range [-128, 128), got -129. Input shape: (10, 3, 128, 64)
    %1 = "ttir.index"(%arg0) <{dim = 2: i32, begin = -129: i32, end = 64: i32, step = 1: i32}> : (tensor<10x3x128x64xbf16>) -> tensor<10x3x64x64xbf16>
    return %1 : tensor<10x3x64x64xbf16>
  }
}

// Verify that the parsing fails if the end value exceeds positive limit
// -----
module attributes {} {
  func.func @index_negative_invalid_end_positive(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16> {
    // CHECK: error: 'ttir.index' op Invalid end index for dimension 1. Expected value in range [-3, 3], got 4. Input shape: (10, 3, 128, 64)
    %1 = "ttir.index"(%arg0) <{dim = 1: i32, begin = 0: i32, end = 4: i32, step = 1: i32}> : (tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16>
    return %1 : tensor<10x3x128x64xbf16>
  }
}

// Verify that the parsing fails if the end value exceeds positive limit
// -----
module attributes {} {
  func.func @index_negative_invalid_end_negative(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16> {
    // CHECK: error: 'ttir.index' op Invalid end index for dimension 1. Expected value in range [-3, 3], got -4. Input shape: (10, 3, 128, 64)
    %1 = "ttir.index"(%arg0) <{dim = 1: i32, begin = -1: i32, end = -4: i32, step = -1: i32}> : (tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16>
    return %1 : tensor<10x3x128x64xbf16>
  }
}

// Verify that the parsing fails if the step value is equal to zero
// -----
module attributes {} {
  func.func @index_negative_step_is_zero(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16> {
    // CHECK: error: 'ttir.index' op Step value for dimension 1 cannot be zero
    %1 = "ttir.index"(%arg0) <{dim = 1: i32, begin = -1: i32, end = -3: i32, step = 0: i32}> : (tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16>
    return %1 : tensor<10x3x128x64xbf16>
  }
}

// Verify that the parsing fails if the begin index is greater than end and step is positive
// -----
module attributes {} {
  func.func @index_negative_begin_greater_than_end_positive_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16> {
    // CHECK: error: 'ttir.index' op For positive step, begin index must be less than or equal to end index for dimension 2. Got begin: 2, end: 0, step: 1, input shape: (10, 3, 128, 64)
    %1 = "ttir.index"(%arg0) <{dim = 2: i32, begin = 2: i32, end = 0: i32, step = 1: i32}> : (tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16>
    return %1 : tensor<10x3x128x64xbf16>
  }
}

// Verify that the parsing fails if the end index is greater than begin and step is negative
// -----
module attributes {} {
  func.func @index_negative_begin_less_than_end_negative_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16> {
    // CHECK: error: 'ttir.index' op For negative step, begin index must be greater than or equal to end index for dimension 3. Got begin: 0, end: 64, step: -1, input shape: (10, 3, 128, 64)
    %1 = "ttir.index"(%arg0) <{dim = 3: i32, begin = 0: i32, end = 64: i32, step = -1: i32}> : (tensor<10x3x128x64xbf16>) -> tensor<10x3x128x64xbf16>
    return %1 : tensor<10x3x128x64xbf16>
  }
}

// Verify that the parsing fails if there is mismatch in output dimension
// -----
module attributes {} {
  func.func @index_negative_invalid_output_shape(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x128x32xbf16> {
    // CHECK: error: 'ttir.index' op Mismatch in dimension 3 of the output tensor: expected size 16, but got 32
    %1 = "ttir.index"(%arg0) <{dim = 3: i32, begin = 0: i32, end = 64: i32, step = 4: i32}> : (tensor<10x3x128x64xbf16>) -> tensor<10x3x128x32xbf16>
    return %1 : tensor<10x3x128x32xbf16>
  }
}
