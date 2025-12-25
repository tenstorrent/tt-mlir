// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// This convolution will not pass if the input is in tile layout. The TTNNWorkaround which sets convolution inputs to row-major should allow this to execute.
module {
  func.func @main(%arg0: tensor<1x128x180x320xbf16>, %arg1: tensor<128x128x3x3xbf16>) -> tensor<1x128x90x160xbf16> {
    %0 = "ttir.conv2d"(%arg0, %arg1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x128x180x320xbf16>, tensor<128x128x3x3xbf16>) -> tensor<1x128x90x160xbf16>
    return %0 : tensor<1x128x90x160xbf16>
  }
}
