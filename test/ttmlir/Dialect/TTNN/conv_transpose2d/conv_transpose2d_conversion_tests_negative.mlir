// RUN: not ttmlir-opt -ttir-to-ttnn-backend-pipeline %s 2>&1 | FileCheck %s

module {
  func.func @forward(%arg0: tensor<20x50x100x16xf32>, %arg1: tensor<16x32x3x3xf32>) -> tensor<20x195x393x32xf32> {
    %0 = ttir.empty() : tensor<20x195x393x32xf32>
    // CHECK: error: 'ttir.conv_transpose2d' op TTNN only supports padding height/width attributes. Thus, padding_top/padding_left must equal padding_bottom/padding_right for the op to execute as expected.
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, output_padding = array<i32: 0, 0>, padding = array<i32: 1, 2, 3, 4>, stride = array<i32: 4, 4>}> : (tensor<20x50x100x16xf32>, tensor<16x32x3x3xf32>, tensor<20x195x393x32xf32>) -> tensor<20x195x393x32xf32>
    return %1 : tensor<20x195x393x32xf32>
  }
}
