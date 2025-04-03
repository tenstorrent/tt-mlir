// RUN: not ttmlir-opt --ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s

// TTNN does not support asymmetric padding
module {
  func.func @conv2d_padding_no_support_for_asymmteric_padding_ttnn(%arg0: tensor<8x32x32x64xbf16>, %arg1: tensor<256x64x3x3xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<8x48x42x256xbf16> {
    %0 = ttir.empty() : tensor<8x48x42x256xbf16>
    // CHECK: error: failed to legalize operation 'ttir.conv2d'
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 12, 8, 6, 4>,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<8x32x32x64xbf16>, tensor<256x64x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<8x48x42x256xbf16>) -> tensor<8x48x42x256xbf16>
    return %1 : tensor<8x48x42x256xbf16>
  }
}
