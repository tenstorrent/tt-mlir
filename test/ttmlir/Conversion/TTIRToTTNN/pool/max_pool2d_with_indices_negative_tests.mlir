// RUN: not ttmlir-opt --split-input-file --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn %s 2>&1 | FileCheck %s

// Test 1: Missing flattened compat info
module {
  func.func @max_pool2d_with_indices_no_flattened_compat(%arg0: tensor<1x128x128x32xbf16>) -> (tensor<1x64x64x32xbf16>, tensor<1x64x64x32xi32>) {
    %0 = ttir.empty() : tensor<1x64x64x32xbf16>
    %1 = ttir.empty() : tensor<1x64x64x32xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op only supports lowering to TTNN for flattened input tensors. Please run the FlattenSlidingWindow pass before lowering to TTNN
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
        kernel = array<i32: 2, 2>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false
    }> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>, tensor<1x64x64x32xi32>) -> (tensor<1x64x64x32xbf16>, tensor<1x64x64x32xi32>)
    return %2, %3 : tensor<1x64x64x32xbf16>, tensor<1x64x64x32xi32>
  }
}
