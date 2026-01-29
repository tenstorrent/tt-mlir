// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module attributes {} {
  func.func @avg_pool2d(%arg0: tensor<1x128x128x32xbf16>) -> (tensor<1x64x64x32xbf16>, tensor<1x64x64x32xbf16>) {
    %1 = "ttir.avg_pool2d"(%arg0) <{kernel = array<i32: 4, 4>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 2, 2>, ceil_mode = false}> : (tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16>
    %3 = "ttir.avg_pool2d"(%arg0) <{kernel = array<i32: 2, 2>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %1, %3 : tensor<1x64x64x32xbf16>, tensor<1x64x64x32xbf16>
  }
}
