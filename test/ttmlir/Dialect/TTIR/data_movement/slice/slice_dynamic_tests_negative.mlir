// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for dynamic slice operation

// Verify that the parsing fails if input is not at least a 1D tensor
module attributes {} {
  func.func @slice_negative_invalid_shape(%arg0: tensor<bf16>, %arg1 : tensor<1xi32>, %arg2 : tensor<1xi32>) -> tensor<1xbf16> {
    %0 = ttir.empty() : tensor<1xbf16>
    // CHECK: error: 'ttir.slice_dynamic' op Input must be at least a 1D tensor
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32]}> : (tensor<bf16>, tensor<1xi32>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1xbf16>
    return %1 : tensor<1xbf16>
  }
}

// Verify that the parsing fails if begins is not a 1D tensor
// -----
module attributes {} {
  func.func @slice_negative_invalid_begins(%arg0: tensor<128x64xbf16>, %arg1 : tensor<2x1xi32>, %arg2 : tensor<2xi32>) -> tensor<64x32xbf16> {
    %0 = ttir.empty() : tensor<64x32xbf16>
    // CHECK: error: 'ttir.slice_dynamic' op Begins and ends must be 1D tensors
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32, 1: i32]}> : (tensor<128x64xbf16>, tensor<2x1xi32>, tensor<2xi32>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %1 : tensor<64x32xbf16>
  }
}

// Verify that the parsing fails if ends is not a 1D tensor
// -----
module attributes {} {
  func.func @slice_negative_invalid_ends(%arg0: tensor<128x64xbf16>, %arg1 : tensor<2xi32>, %arg2 : tensor<2x1xi32>) -> tensor<64x32xbf16> {
    %0 = ttir.empty() : tensor<64x32xbf16>
    // CHECK: error: 'ttir.slice_dynamic' op Begins and ends must be 1D tensors
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32, 1: i32]}> : (tensor<128x64xbf16>, tensor<2xi32>, tensor<2x1xi32>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %1 : tensor<64x32xbf16>
  }
}

// Verify that the parsing fails if the ends size is not equal to the input tensor rank
// -----
module attributes {} {
  func.func @slice_negative_invalid_ends_shape(%arg0: tensor<128x64xbf16>, %arg1 : tensor<2xi32>, %arg2 : tensor<3xi32>) -> tensor<64x32xbf16> {
    %0 = ttir.empty() : tensor<64x32xbf16>
    // CHECK: error: 'ttir.slice_dynamic' op Begins, ends, and step must have the same number of elements as the input tensor rank
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32, 1: i32]}> : (tensor<128x64xbf16>, tensor<2xi32>, tensor<3xi32>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %1 : tensor<64x32xbf16>
  }
}

// Verify that the parsing fails if the step size is not equal to the input tensor rank
// -----
module attributes {} {
  func.func @slice_negative_invalid_step(%arg0: tensor<128x64xbf16>, %arg1 : tensor<2xi32>, %arg2 : tensor<2xi32>) -> tensor<64x32xbf16> {
    %0 = ttir.empty() : tensor<64x32xbf16>
    // CHECK: error: 'ttir.slice_dynamic' op Begins, ends, and step must have the same number of elements as the input tensor rank
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32]}> : (tensor<128x64xbf16>, tensor<2xi32>, tensor<2xi32>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %1 : tensor<64x32xbf16>
  }
}

// Verify that the parsing fails if the output type is not equal to the input tensor type
// -----
module attributes {} {
  func.func @slice_negative_invalid_output_datatype(%arg0: tensor<128x64xbf16>, %arg1 : tensor<2xi32>, %arg2 : tensor<2xi32>) -> tensor<64x32xf32> {
    %0 = ttir.empty() : tensor<64x32xf32>
    // CHECK: error: 'ttir.slice_dynamic' op Output tensor must have the same element type as the input tensor
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32, 1: i32]}> : (tensor<128x64xbf16>, tensor<2xi32>, tensor<2xi32>, tensor<64x32xf32>) -> tensor<64x32xf32>
    return %1 : tensor<64x32xf32>
  }
}

// Verify that the parsing fails if the output rank is not equal to the input tensor rank
// -----
module attributes {} {
  func.func @slice_negative_input_output_rank_missmatch(%arg0: tensor<128x64xbf16>, %arg1 : tensor<2xi32>, %arg2 : tensor<2xi32>) -> tensor<64x32x1xbf16> {
    %0 = ttir.empty() : tensor<64x32x1xbf16>
    // CHECK: error: 'ttir.slice_dynamic' op Output tensor must have the same rank as the input tensor
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32, 1: i32]}> : (tensor<128x64xbf16>, tensor<2xi32>, tensor<2xi32>, tensor<64x32x1xbf16>) -> tensor<64x32x1xbf16>
    return %1 : tensor<64x32x1xbf16>
  }
}

// Verify that the parsing fails if the step value is equal to zero
// -----
module attributes {} {
  func.func @slice_negative_step_is_zero(%arg0: tensor<128x64xbf16>, %arg1 : tensor<2xi32>, %arg2 : tensor<2xi32>) -> tensor<64x32xbf16> {
    %0 = ttir.empty() : tensor<64x32xbf16>
    // CHECK: error: 'ttir.slice_dynamic' op Step value for dimension 1 cannot be zero
    %1 = "ttir.slice_dynamic"(%arg0, %arg1, %arg2, %0) <{step = [1: i32, 0: i32]}> : (tensor<128x64xbf16>, tensor<2xi32>, tensor<2xi32>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %1 : tensor<64x32xbf16>
  }
}
