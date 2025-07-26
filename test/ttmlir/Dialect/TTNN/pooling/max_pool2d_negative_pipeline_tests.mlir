// RUN: not ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s
// Negative tests for max_pool2d pipeline

// Test 1: MaxPool2dOp with asymmetric padding top/bottom
module {
  func.func @max_pool2d_invalid_padding_horizontal(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: error: 'ttir.max_pool2d' op only supports lowering to TTNN for symmetric padding for top/bottom
    %1 = "ttir.max_pool2d"(%arg0, %0) <{
      kernel = array<i32: 4, 4>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// -----

// Test 2: MaxPool2dOp with asymmetric padding left/right
module {
  func.func @max_pool2d_invalid_padding_vertical(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x29x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x29x30x64xbf16>
    // CHECK: error: 'ttir.max_pool2d' op only supports lowering to TTNN for symmetric padding for left/right
    %1 = "ttir.max_pool2d"(%arg0, %0) <{
      kernel = array<i32: 4, 4>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 1, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x29x30x64xbf16>) -> tensor<1x29x30x64xbf16>
    return %1 : tensor<1x29x30x64xbf16>
  }
}
