// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module attributes {} {
  func.func @forward(%arg0: tensor<4x32x32xbf16>) -> tensor<2x16x16xbf16> {
    %0 = ttir.empty() : tensor<2x16x16xbf16>
    // CHECK: = "ttnn.slice_static"
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>, tensor<2x16x16xbf16>) -> tensor<2x16x16xbf16>
    return %1 : tensor<2x16x16xbf16>
  }

  func.func @test_slice_empty(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x8x0xbf16> {
    // CHECK-LABEL: @test_slice_empty
    %0 = ttir.empty() : tensor<4x1x8x0xbf16>
    // CHECK: %{{[0-9]+}} = "ttnn.slice_static"
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0: i32, 0: i32, 32: i32, 128: i32], ends = [10: i32, 3: i32, 64: i32, 128: i32], step = [3: i32, 3: i32, 4: i32, 8: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x8x0xbf16>) -> tensor<4x1x8x0xbf16>
    return %1 : tensor<4x1x8x0xbf16>
  }

  func.func @test_slice_f32(%arg0: tensor<4x32x32xf32>) -> tensor<2x16x16xf32> {
    %0 = ttir.empty() : tensor<2x16x16xf32>
    // CHECK: = "ttnn.slice_static"
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xf32>, tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
    return %1 : tensor<2x16x16xf32>
  }

  func.func @test_slice_f16(%arg0: tensor<4x32x32xf16>) -> tensor<2x16x16xf16> {
    %0 = ttir.empty() : tensor<2x16x16xf16>
    // CHECK: = "ttnn.slice_static"
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xf16>, tensor<2x16x16xf16>) -> tensor<2x16x16xf16>
    return %1 : tensor<2x16x16xf16>
  }

  func.func @test_slice_uint32(%arg0: tensor<4x32x32xui32>) -> tensor<2x16x16xui32> {
    %0 = ttir.empty() : tensor<2x16x16xui32>
    // CHECK: = "ttnn.slice_static"
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xui32>, tensor<2x16x16xui32>) -> tensor<2x16x16xui32>
    return %1 : tensor<2x16x16xui32>
  }

  func.func @test_slice_uint16(%arg0: tensor<4x32x32xui16>) -> tensor<2x16x16xui16> {
    %0 = ttir.empty() : tensor<2x16x16xui16>
    // CHECK: = "ttnn.slice_static"
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xui16>, tensor<2x16x16xui16>) -> tensor<2x16x16xui16>
    return %1 : tensor<2x16x16xui16>
  }

  func.func @test_slice_uint8(%arg0: tensor<4x32x32xui8>) -> tensor<2x16x16xui8> {
    %0 = ttir.empty() : tensor<2x16x16xui8>
    // CHECK: = "ttnn.slice_static"
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xui8>, tensor<2x16x16xui8>) -> tensor<2x16x16xui8>
    return %1 : tensor<2x16x16xui8>
  }

  func.func @test_slice_int32(%arg0: tensor<4x32x32xi32>) -> tensor<2x16x16xi32> {
    %0 = ttir.empty() : tensor<2x16x16xi32>
    // CHECK: = "ttnn.slice_static"
    %1 = "ttir.slice_static"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xi32>, tensor<2x16x16xi32>) -> tensor<2x16x16xi32>
    return %1 : tensor<2x16x16xi32>
  }
}
