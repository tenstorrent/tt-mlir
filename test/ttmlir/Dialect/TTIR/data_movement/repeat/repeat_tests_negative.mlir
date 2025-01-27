// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for repeat operation

// Verify that the parsing fails if the input tensor and repeat_dimensions attribute doesn't have the same rank
module {
  func.func @repeat_not_valid_repeat_dimension_attribute(%arg0: tensor<32x32xf32>) -> tensor<32x64xf32> {
    // CHECK: 'ttir.repeat' op Input tensor rank 2 doesn't match the number of repeat dimensions 1.
    %0 = tensor.empty() : tensor<32x64xf32>
    %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i64 : 2>} : (tensor<32x32xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }
}

// -----

// Verify that the parsing fails if the input tensor and repeat_dimensions attribute doesn't have the same rank
module {
  func.func @repeat_not_valid_input_output(%arg0: tensor<32x32xf32>) -> tensor<1x32x64xf32> {
    // CHECK: 'ttir.repeat' op Input tensor rank 2 doesn't match the output tensor rank 3.
    %0 = tensor.empty() : tensor<1x32x64xf32>
    %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i64 : 1, 2>} : (tensor<32x32xf32>, tensor<1x32x64xf32>) -> tensor<1x32x64xf32>
    return %1 : tensor<1x32x64xf32>
  }
}

// -----

// Verify that the parsing fails if the output tensor dimensions are not expected
module {
  func.func @repeat_not_valid_input_output(%arg0: tensor<32x32xf32>) -> tensor<32x128xf32> {
    // CHECK: 'ttir.repeat' op Input tensor shape (32,32) at index 1 does not repeat to output (32,128) using repeat value 2.
    %0 = tensor.empty() : tensor<32x128xf32>
    %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i64 : 1, 2>} : (tensor<32x32xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }
}

// -----

// Verify that the parsing fails if there is a negative repeat dimension
module {
  func.func @repeat_not_valid_input_output(%arg0: tensor<32x32xf32>) -> tensor<32x128xf32> {
    // CHECK: ttir.repeat' op Repeat dimension at index 0 must be greater than 0.
    %0 = tensor.empty() : tensor<32x128xf32>
    %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i64 : -1, 2>} : (tensor<32x32xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }
}
