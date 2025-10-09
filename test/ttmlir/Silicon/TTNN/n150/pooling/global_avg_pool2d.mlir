// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  // Test 1: Basic GlobalAvgPool2d with square input
  func.func @global_avg_pool2d_basic_square(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x1x1x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x64xbf16>
    // CHECK: ttnn.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x32x32x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
    return %1 : tensor<1x1x1x64xbf16>
  }

  // Test 2: GlobalAvgPool2d with rectangular input (different H and W)
  func.func @global_avg_pool2d_rectangular(%arg0: tensor<1x64x32x128xbf16>) -> tensor<1x1x1x128xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x128xbf16>
    // CHECK: ttnn.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x64x32x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    return %1 : tensor<1x1x1x128xbf16>
  }

  // Test 3: GlobalAvgPool2d with larger spatial dimensions
  func.func @global_avg_pool2d_large_spatial(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x1x1x32xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x32xbf16>
    // CHECK: ttnn.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x128x128x32xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xbf16>
    return %1 : tensor<1x1x1x32xbf16>
  }

  // Test 4: GlobalAvgPool2d with different batch size
  func.func @global_avg_pool2d_batch_size(%arg0: tensor<4x16x16x256xbf16>) -> tensor<4x1x1x256xbf16> {
    %0 = ttir.empty() : tensor<4x1x1x256xbf16>
    // CHECK: ttnn.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<4x16x16x256xbf16>, tensor<4x1x1x256xbf16>) -> tensor<4x1x1x256xbf16>
    return %1 : tensor<4x1x1x256xbf16>
  }

  // Test 5: GlobalAvgPool2d with f32 data type
  func.func @global_avg_pool2d_f32(%arg0: tensor<1x56x56x64xf32>) -> tensor<1x1x1x64xf32> {
    %0 = ttir.empty() : tensor<1x1x1x64xf32>
    // CHECK: ttnn.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>
    return %1 : tensor<1x1x1x64xf32>
  }

  // Test 6: GlobalAvgPool2d with very large channel count (ResNet-like)
  func.func @global_avg_pool2d_large_channels(%arg0: tensor<1x14x14x2048xbf16>) -> tensor<1x1x1x2048xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x2048xbf16>
    // CHECK: ttnn.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x14x14x2048xbf16>, tensor<1x1x1x2048xbf16>) -> tensor<1x1x1x2048xbf16>
    return %1 : tensor<1x1x1x2048xbf16>
  }

  // Test 7: GlobalAvgPool2d with minimal spatial dimensions (edge case)
  func.func @global_avg_pool2d_minimal_spatial(%arg0: tensor<1x2x2x64xbf16>) -> tensor<1x1x1x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x64xbf16>
    // CHECK: ttnn.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x2x2x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
    return %1 : tensor<1x1x1x64xbf16>
  }
}
