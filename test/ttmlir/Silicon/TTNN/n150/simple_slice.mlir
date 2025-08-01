// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module attributes {} {
  func.func @forward(%arg0: tensor<4x32x32xbf16>) -> tensor<2x16x16xbf16> {
    %0 = ttir.empty() : tensor<2x16x16xbf16>
    // CHECK: = "ttnn.slice"
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>, tensor<2x16x16xbf16>) -> tensor<2x16x16xbf16>
    return %1 : tensor<2x16x16xbf16>
  }

  func.func @test_slice_empty(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x8x0xbf16> {
    // CHECK-LABEL: @test_slice_empty
    %0 = ttir.empty() : tensor<4x1x8x0xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.slice"
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 32: i32, 128: i32], ends = [10: i32, 3: i32, 64: i32, 128: i32], step = [3: i32, 3: i32, 4: i32, 8: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x8x0xbf16>) -> tensor<4x1x8x0xbf16>
    return %1 : tensor<4x1x8x0xbf16>
  }
}
