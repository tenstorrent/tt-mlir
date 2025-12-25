
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// This convolution will not pass if the input is in tile layout. The TTNNWorkaround which sets convolution inputs to row-major should allow this to execute.
module {
  func.func @main(%arg0: tensor<1x256x180x320xbf16>, %arg1: tensor<512x256x1x1xbf16>) -> tensor<1x512x90x160xbf16> {
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x256x180x320xbf16>) -> tensor<1x180x320x256xbf16>
    %1 = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<512x256x1x1xbf16>) -> tensor<512x256x1x1xbf16>
    %2 = "ttir.conv2d"(%0, %1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>}> : (tensor<1x180x320x256xbf16>, tensor<512x256x1x1xbf16>) -> tensor<1x90x160x512xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x90x160x512xbf16>) -> tensor<1x512x90x160xbf16>
    return %3 : tensor<1x512x90x160xbf16>
  }
}
