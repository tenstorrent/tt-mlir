// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for repeat operation

// Verify that the parsing fails if the input tensor and repeat_dims attribute doesn't have the same rank
module {
  func.func @repeat_not_valid_repeat_dimension_attribute(%arg0: tensor<32x32xf32>) -> tensor<32x64xf32> {
    // CHECK: 'ttnn.repeat' op Input tensor rank 2 doesn't match the number of repeat dimensions 1.
    %0 = "ttnn.repeat"(%arg0) {repeat_dims = #ttnn.shape<2>} : (tensor<32x32xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
  }
}

// -----

// Verify that the parsing fails if the input tensor and repeat_dims attribute doesn't have the same rank
module {
  func.func @repeat_not_valid_input_output(%arg0: tensor<32x32xf32>) -> tensor<1x32x64xf32> {
    // CHECK: 'ttnn.repeat' op Input tensor rank 2 doesn't match the output tensor rank 3.
    %0 = "ttnn.repeat"(%arg0) {repeat_dims = #ttnn.shape<1x2>} : (tensor<32x32xf32>) -> tensor<1x32x64xf32>
    return %0 : tensor<1x32x64xf32>
  }
}

// -----

// Verify that the parsing fails if the output tensor dimensions are not expected
module {
  func.func @repeat_not_valid_input_output(%arg0: tensor<32x32xf32>) -> tensor<32x128xf32> {
    // CHECK: 'ttnn.repeat' op Input tensor shape (32,32) at index 1 does not repeat to output (32,128) using repeat value 2.
    %0 = "ttnn.repeat"(%arg0) {repeat_dims = #ttnn.shape<1x2>} : (tensor<32x32xf32>) -> tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }
}
