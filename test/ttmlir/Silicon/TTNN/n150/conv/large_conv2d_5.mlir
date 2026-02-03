// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// This convolution will not pass if the input is in tile layout. The TTNNWorkaround which sets convolution inputs to row-major should allow this to execute.
module {
  func.func @main(%arg0: tensor<1x3x224x224xbf16>, %arg1: tensor<1024x3x16x16xbf16>, %arg2: tensor<1024xbf16>) -> tensor<1x1024x14x14xbf16> {
    %0 = "ttir.conv2d"(%arg0, %arg1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 16, 16>, batch_dim = 0 : i64, channel_dim = 1 : i64, height_dim = 2 : i64, width_dim = 3 : i64}> : (tensor<1x3x224x224xbf16>, tensor<1024x3x16x16xbf16>) -> tensor<1x1024x14x14xbf16>
    %1 = "ttir.reshape"(%arg2) <{shape = [1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024xbf16>) -> tensor<1024x1x1xbf16>
    %2 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    %3 = "ttir.reshape"(%1) <{shape = [1 : i32, 1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1024x1x1xbf16>) -> tensor<1x1024x1x1xbf16>
    %4 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 14, 14>}> : (tensor<1x1024x1x1xbf16>) -> tensor<1x1024x14x14xbf16>
    %5 = "ttir.add"(%0, %4) : (tensor<1x1024x14x14xbf16>, tensor<1x1024x14x14xbf16>) -> tensor<1x1024x14x14xbf16>
    return %5 : tensor<1x1024x14x14xbf16>
  }
}
