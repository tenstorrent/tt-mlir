// RUN: ttmlir-opt %s | FileCheck %s
module attributes {} {
  func.func @slice_1d(%arg0: tensor<64xbf16>) -> tensor<32xbf16> {
    %0 = tensor.empty() : tensor<32xbf16>
    // CHECK: %[[C:.*]] = "ttir.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32], ends = [32: i32], step = [1: i32]}> : (tensor<64xbf16>, tensor<32xbf16>) -> tensor<32xbf16>
    return %1 : tensor<32xbf16>
  }

  func.func @slice_1d_step(%arg0: tensor<64xbf16>) -> tensor<16xbf16> {
    %0 = tensor.empty() : tensor<16xbf16>
    // CHECK: %[[C:.*]] = "ttir.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32], ends = [64: i32], step = [4: i32]}> : (tensor<64xbf16>, tensor<16xbf16>) -> tensor<16xbf16>
    return %1 : tensor<16xbf16>
  }

  func.func @slice_2d(%arg0: tensor<128x64xbf16>) -> tensor<64x32xbf16> {
    %0 = tensor.empty() : tensor<64x32xbf16>
    // CHECK: %[[C:.*]] = "ttir.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32], ends = [64: i32, 32: i32], step = [1: i32, 1: i32]}> : (tensor<128x64xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %1 : tensor<64x32xbf16>
  }

  func.func @slice_2d_step(%arg0: tensor<128x64xbf16>) -> tensor<32x16xbf16> {
    %0 = tensor.empty() : tensor<32x16xbf16>
    // CHECK: %[[C:.*]] = "ttir.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [64: i32, 0: i32], ends = [128: i32, 64: i32], step = [2: i32, 4: i32]}> : (tensor<128x64xbf16>, tensor<32x16xbf16>) -> tensor<32x16xbf16>
    return %1 : tensor<32x16xbf16>
  }

  func.func @slice_3d(%arg0: tensor<3x128x64xbf16>) -> tensor<1x64x64xbf16> {
    %0 = tensor.empty() : tensor<1x64x64xbf16>
    // CHECK: %[[C:.*]] = "ttir.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [1: i32, 64: i32, 64: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<3x128x64xbf16>, tensor<1x64x64xbf16>) -> tensor<1x64x64xbf16>
    return %1 : tensor<1x64x64xbf16>
  }

  func.func @slice_3d_step(%arg0: tensor<3x128x64xbf16>) -> tensor<2x32x32xbf16> {
    %0 = tensor.empty() : tensor<2x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttir.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 32: i32], ends = [3: i32, 128: i32, 64: i32], step = [2: i32, 4: i32, 1: i32]}> : (tensor<3x128x64xbf16>, tensor<2x32x32xbf16>) -> tensor<2x32x32xbf16>
    return %1 : tensor<2x32x32xbf16>
  }

  func.func @slice_4d(%arg0: tensor<10x3x128x64xbf16>) -> tensor<5x3x32x64xbf16> {
    %0 = tensor.empty() : tensor<5x3x32x64xbf16>
    // CHECK: %[[C:.*]] = "ttir.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [3: i32, 0: i32, 32: i32, 0: i32], ends = [8: i32, 3: i32, 64: i32, 64: i32], step = [1: i32, 1: i32, 1: i32, 1: i32]}> : (tensor<10x3x128x64xbf16>, tensor<5x3x32x64xbf16>) -> tensor<5x3x32x64xbf16>
    return %1 : tensor<5x3x32x64xbf16>
  }

  func.func @slice_4d_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x32xbf16> {
    %0 = tensor.empty() : tensor<4x1x16x32xbf16>
    // CHECK: %[[C:.*]] = "ttir.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 2: i32, 0: i32, -64: i32], ends = [10: i32, 0: i32, -1: i32, -1: i32], step = [3: i32, -2: i32, 8: i32, 2: i32]}> : (tensor<10x3x128x64xbf16>, tensor<4x1x16x32xbf16>) -> tensor<4x1x16x32xbf16>
    return %1 : tensor<4x1x16x32xbf16>
  }
}
