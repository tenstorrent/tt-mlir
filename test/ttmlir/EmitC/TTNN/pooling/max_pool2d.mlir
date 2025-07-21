// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module attributes {} {
  func.func @max_pool2d(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
    %0 = ttir.empty() : tensor<1x64x64x32xbf16>
    %1 = "ttir.max_pool2d"(%arg0, %0) <{kernel_height=2: si32, kernel_width=2: si32, stride_height=2: si32, stride_width=2: si32, dilation_height=1: si32, dilation_width=1: si32, ceil_mode=false, padding_left=0: si32, padding_right=0: si32, padding_top=0: si32, padding_bottom=0: si32}> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %1 : tensor<1x64x64x32xbf16>
  }
}
