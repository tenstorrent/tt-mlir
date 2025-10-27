// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module attributes {} {
  func.func @forward(%arg0: tensor<1x128x128x32xbf16>) -> (tensor<1x64x64x32xbf16>, tensor<1x64x64x32xsi32>) {
    %0 = ttir.empty() : tensor<1x64x64x32xbf16>
    %1 = ttir.empty() : tensor<1x64x64x32xsi32>
    // CHECK: = "ttnn.max_pool2d_with_indices"
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{kernel = array<i32: 2, 2>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x128x128x32xbf16>, tensor<1x64x64x32xbf16>, tensor<1x64x64x32xsi32>) -> (tensor<1x64x64x32xbf16>, tensor<1x64x64x32xsi32>)
    return %2, %3 : tensor<1x64x64x32xbf16>, tensor<1x64x64x32xsi32>
  }
}
