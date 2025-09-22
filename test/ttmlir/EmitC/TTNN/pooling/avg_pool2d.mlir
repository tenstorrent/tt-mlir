// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module attributes {} {
  func.func @avg_pool2d(%arg0: tensor<1x128x128x32xbf16>) -> (tensor<1x64x64x32xbf16>, tensor<1x64x64x32xbf16>) {
    %0 = ttir.empty() : tensor<1x64x64x32xbf16>
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{kernel = array<i32: 4, 4>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 2, 2>, ceil_mode = false}> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>) -> tensor<1x64x64x32xbf16>
    %2 = ttir.empty() : tensor<1x64x64x32xbf16>
    %3 = "ttir.avg_pool2d"(%arg0, %2) <{kernel = array<i32: 2, 2>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %1, %3 : tensor<1x64x64x32xbf16>, tensor<1x64x64x32xbf16>
  }
}
