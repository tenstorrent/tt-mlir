
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// This convolution will not pass if the input is in tile layout. The TTNNWorkaround which sets convolution inputs to row-major should allow this to execute.
module {
  func.func @main(%arg0: tensor<1x3x518x518xbf16>, %arg1: tensor<1280x3x14x14xbf16>, %arg2: tensor<1280xbf16>) -> tensor<1x1280x37x37xbf16> {
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x3x518x518xbf16>) -> tensor<1x518x518x3xbf16>
    %1 = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1280x3x14x14xbf16>) -> tensor<1280x3x14x14xbf16>
    %2 = "ttir.conv2d"(%0, %1) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 14, 14>}> : (tensor<1x518x518x3xbf16>, tensor<1280x3x14x14xbf16>) -> tensor<1x37x37x1280xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x37x37x1280xbf16>) -> tensor<1x1280x37x37xbf16>
    %4 = "ttir.reshape"(%arg2) <{shape = [1280 : i32, 1 : i32, 1 : i32]}> : (tensor<1280xbf16>) -> tensor<1280x1x1xbf16>
    %5 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 1, 1, 1, 1>}> : (tensor<1x1280x37x37xbf16>) -> tensor<1x1280x37x37xbf16>
    %6 = "ttir.reshape"(%4) <{shape = [1 : i32, 1280 : i32, 1 : i32, 1 : i32]}> : (tensor<1280x1x1xbf16>) -> tensor<1x1280x1x1xbf16>
    %7 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 1, 1, 37, 37>}> : (tensor<1x1280x1x1xbf16>) -> tensor<1x1280x37x37xbf16>
    %8 = "ttir.add"(%5, %7) : (tensor<1x1280x37x37xbf16>, tensor<1x1280x37x37xbf16>) -> tensor<1x1280x37x37xbf16>
    return %8 : tensor<1x1280x37x37xbf16>
  }
}
