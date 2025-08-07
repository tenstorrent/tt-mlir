// RUN: not ttmlir-opt --split-input-file --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn %s 2>&1 | FileCheck %s

// Test 1: Missing flattened compat info
module {
  func.func @avg_pool2d_no_flattened_compat(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
    %0 = ttir.empty() : tensor<1x64x64x32xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op only supports lowering to TTNN for flattened input tensors. Please run the FlattenSlidingWindow pass before lowering to TTNN
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
        kernel = array<i32: 2, 2>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false
    }> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %1 : tensor<1x64x64x32xbf16>
  }
}

// -----
// Test 2: Test that for avg_pool2d, dilation is not supported
module {
  func.func @avg_pool2d_with_dilation(%arg0: tensor<1x1x16384x32xbf16>) -> tensor<1x1x3969x32xbf16> {
    %0 = ttir.empty() : tensor<1x1x3969x32xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op only supports lowering to TTNN for dilation of (1, 1)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
        kernel = array<i32: 2, 2>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 3, 3>, // Dilation is not supported
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false,
        flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 128, input_width = 128>
    }> : (tensor<1x1x16384x32xbf16>, tensor<1x1x3969x32xbf16>) -> tensor<1x1x3969x32xbf16>
    return %1 : tensor<1x1x3969x32xbf16>
  }
}
