// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for reshape operation

// Verify that verification fails when shape size doesn't matches the rank of the output tensor
module {
  func.func @reshape_shape_size_different_from_the_output_rank(%arg0: tensor<2x32x32xbf16>) -> tensor<1x32x2x32xbf16> {
    %1 = "ttnn.reshape"(%arg0) <{shape = [32: i32, 2: i32, 32: i32]}> : (tensor<2x32x32xbf16>) -> tensor<1x32x2x32xbf16>
    // CHECK: error: 'ttnn.reshape' op Shape attribute size 3 must match output tensor rank 4
    return %1 : tensor<1x32x2x32xbf16>
  }
}

// -----
// Verify that verification fails when input and output tensor have different number of elements.
module {
  func.func @reshape_input_output_elements_mismatch(%arg0: tensor<2x32x32xbf16>) -> tensor<32x3x32xbf16> {
    %1 = "ttnn.reshape"(%arg0) <{shape = [32: i32, 3: i32, 32: i32]}> : (tensor<2x32x32xbf16>) -> tensor<32x3x32xbf16>
    // CHECK: error: 'ttnn.reshape' op Input tensor number of elements 2048 and output tensor number of elements 3072 must be the same
    return %1 : tensor<32x3x32xbf16>
  }
}

// -----
// Verify that verification fails when shape attribute has more than one -1(infer) element.
module {
  func.func @reshape_infer_dim_negative(%arg0: tensor<2x32x32xbf16>) -> tensor<32x2x32xbf16> {
    %1 = "ttnn.reshape"(%arg0) <{shape = [32: i32, -1: i32, -1: i32]}> : (tensor<2x32x32xbf16>) -> tensor<32x2x32xbf16>
    // CHECK: error: 'ttnn.reshape' op Shape attribute must have at most one -1 element
    return %1 : tensor<32x2x32xbf16>
  }
}

// -----
// Verify that verification fails if the shape attribute has negative dimension which is not -1.
module {
  func.func @reshape_infer_dim_negative(%arg0: tensor<2x32x32xbf16>) -> tensor<32x2x32xbf16> {
    %1 = "ttnn.reshape"(%arg0) <{shape = [32: i32, -1: i32, -32: i32]}> : (tensor<2x32x32xbf16>) -> tensor<32x2x32xbf16>
    // CHECK: error: 'ttnn.reshape' op All dimensions must be >= 0 except the one with -1
    return %1 : tensor<32x2x32xbf16>
  }
}

// -----
// Verify that verification fails if the shape attribute is different from the output tensor shape.
module {
  func.func @reshape_shape_mismatch(%arg0: tensor<2x32x32xbf16>) -> tensor<32x2x32xbf16> {
    %1 = "ttnn.reshape"(%arg0) <{shape = [32: i32, 3: i32, 32: i32]}> : (tensor<2x32x32xbf16>) -> tensor<32x2x32xbf16>
    // CHECK: error: 'ttnn.reshape' op Shape attribute 3 must match the output tensor shape 2 at index 1 for dimension that is not -1
    return %1 : tensor<32x2x32xbf16>
  }
}
