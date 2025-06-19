// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline="target-dylib=true" %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module attributes {} {
  func.func @max_pool2d(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
    %0 = ttir.empty() : tensor<1x64x64x32xbf16>
    %1 = "ttir.max_pool2d"(%arg0, %0) <{kernel_height=2: si32, kernel_width=2: si32, stride_height=2: si32, stride_width=2: si32, dilation_height=1: si32, dilation_width=1: si32, ceil_mode=false, padding_left=0: si32, padding_right=0: si32, padding_top=0: si32, padding_bottom=0: si32}> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %1 : tensor<1x64x64x32xbf16>
  }
}
