// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// This convolution will not pass if the input is in tile layout. The TTNNWorkaround which sets convolution inputs to row-major should allow this to execute.
module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<768x3x16x16xbf16>, %arg2: tensor<768xbf16>) -> tensor<1x768x14x14xbf16> {
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x224x224xbf16>) -> tensor<1x224x224x3xbf16>
    %1 = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<768x3x16x16xbf16>) -> tensor<768x3x16x16xbf16>
    %2 = "ttir.conv2d"(%0, %1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 16, 16>}> : (tensor<1x224x224x3xbf16>, tensor<768x3x16x16xbf16>) -> tensor<1x14x14x768xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x14x14x768xbf16>) -> tensor<1x768x14x14xbf16>
    %4 = "ttir.reshape"(%arg2) <{shape = [768 : i32, 1 : i32, 1 : i32]}> : (tensor<768xbf16>) -> tensor<768x1x1xbf16>
    %5 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x768x14x14xbf16>) -> tensor<1x768x14x14xbf16>
    %6 = "ttir.reshape"(%4) <{shape = [1 : i32, 768 : i32, 1 : i32, 1 : i32]}> : (tensor<768x1x1xbf16>) -> tensor<1x768x1x1xbf16>
    %7 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 1, 1, 14, 14>}> : (tensor<1x768x1x1xbf16>) -> tensor<1x768x14x14xbf16>
    %8 = "ttir.add"(%5, %7) : (tensor<1x768x14x14xbf16>, tensor<1x768x14x14xbf16>) -> tensor<1x768x14x14xbf16>
    return %8 : tensor<1x768x14x14xbf16>
  }
}
