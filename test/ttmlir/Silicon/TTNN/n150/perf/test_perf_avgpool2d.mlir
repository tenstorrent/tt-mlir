// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// UNSUPPORTED: true
module attributes {} {
  func.func @test_avgpool2d(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
    // CHECK-LABEL: @test_avgpool2d
    %0 = ttir.empty() : tensor<1x64x64x32xbf16>
    // CHECK: = "ttnn.avg_pool2d"
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{kernel = array<i32: 2, 2>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, ceil_mode=false, padding = array<i32: 0, 0, 0, 0>}> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>) -> tensor<1x64x64x32xbf16>
    return %1 : tensor<1x64x64x32xbf16>
  }
}
