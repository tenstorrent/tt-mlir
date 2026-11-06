// RUN: ttmlir-opt -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @slice_1d(%arg0: tensor<64xbf16>) -> tensor<32xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [0: i32], ends = [32: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<32xbf16>
    return %1 : tensor<32xbf16>
  }

  func.func @slice_1d_empty_result_case_01(%arg0: tensor<64xbf16>) -> tensor<0xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [3: i32], ends = [1: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<0xbf16>
    return %1 : tensor<0xbf16>
  }

  func.func @slice_1d_empty_result_case_02(%arg0: tensor<64xbf16>) -> tensor<0xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [-92: i32], ends = [-90: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<0xbf16>
    return %1 : tensor<0xbf16>
  }

  func.func @slice_1d_empty_result_case_03(%arg0: tensor<64xbf16>) -> tensor<0xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [-90: i32], ends = [-92: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<0xbf16>
    return %1 : tensor<0xbf16>
  }

  func.func @slice_1d_empty_result_case_04(%arg0: tensor<64xbf16>) -> tensor<0xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [66: i32], ends = [68: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<0xbf16>
    return %1 : tensor<0xbf16>
  }

  func.func @slice_1d_empty_result_case_05(%arg0: tensor<64xbf16>) -> tensor<0xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [68: i32], ends = [66: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<0xbf16>
    return %1 : tensor<0xbf16>
  }

  func.func @slice_1d_empty_result_case_06(%arg0: tensor<64xbf16>) -> tensor<0xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [64: i32], ends = [64: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<0xbf16>
    return %1 : tensor<0xbf16>
  }

  func.func @slice_1d_out_of_bounds_indexing_case_01(%arg0: tensor<64xbf16>) -> tensor<32xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [32: i32], ends = [100: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<32xbf16>
    return %1 : tensor<32xbf16>
  }

  func.func @slice_1d_out_of_bounds_indexing_case_02(%arg0: tensor<64xbf16>) -> tensor<64xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [-124: i32], ends = [124: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<64xbf16>
    return %1 : tensor<64xbf16>
  }

  func.func @slice_1d_out_of_bounds_indexing_case_03(%arg0: tensor<64xbf16>) -> tensor<32xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [-200: i32], ends = [32: i32], step = [1: i32]}> : (tensor<64xbf16>) -> tensor<32xbf16>
    return %1 : tensor<32xbf16>
  }

  func.func @slice_1d_step(%arg0: tensor<64xbf16>) -> tensor<16xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [0: i32], ends = [64: i32], step = [4: i32]}> : (tensor<64xbf16>) -> tensor<16xbf16>
    return %1 : tensor<16xbf16>
  }

  func.func @slice_2d(%arg0: tensor<128x64xbf16>) -> tensor<64x32xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [0: i32, 0: i32], ends = [64: i32, 32: i32], step = [1: i32, 1: i32]}> : (tensor<128x64xbf16>) -> tensor<64x32xbf16>
    return %1 : tensor<64x32xbf16>
  }

  func.func @slice_2d_step(%arg0: tensor<128x64xbf16>) -> tensor<32x16xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [64: i32, 0: i32], ends = [128: i32, 64: i32], step = [2: i32, 4: i32]}> : (tensor<128x64xbf16>) -> tensor<32x16xbf16>
    return %1 : tensor<32x16xbf16>
  }

  func.func @slice_3d(%arg0: tensor<3x128x64xbf16>) -> tensor<1x64x64xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [0: i32, 0: i32, 0: i32], ends = [1: i32, 64: i32, 64: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<3x128x64xbf16>) -> tensor<1x64x64xbf16>
    return %1 : tensor<1x64x64xbf16>
  }

  func.func @slice_3d_step(%arg0: tensor<3x128x64xbf16>) -> tensor<2x32x32xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [0: i32, 0: i32, 32: i32], ends = [3: i32, 128: i32, 64: i32], step = [2: i32, 4: i32, 1: i32]}> : (tensor<3x128x64xbf16>) -> tensor<2x32x32xbf16>
    return %1 : tensor<2x32x32xbf16>
  }

  func.func @slice_4d(%arg0: tensor<10x3x128x64xbf16>) -> tensor<5x3x32x64xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [3: i32, 0: i32, 32: i32, 0: i32], ends = [8: i32, 3: i32, 64: i32, 64: i32], step = [1: i32, 1: i32, 1: i32, 1: i32]}> : (tensor<10x3x128x64xbf16>) -> tensor<5x3x32x64xbf16>
    return %1 : tensor<5x3x32x64xbf16>
  }

  func.func @slice_4d_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<4x1x16x32xbf16> {
    // CHECK: = "ttir.slice_static"
    %1 = "ttir.slice_static"(%arg0) <{begins = [0: i32, 2: i32, 0: i32, -64: i32], ends = [10: i32, 0: i32, -1: i32, -1: i32], step = [3: i32, -2: i32, 8: i32, 2: i32]}> : (tensor<10x3x128x64xbf16>) -> tensor<4x1x16x32xbf16>
    return %1 : tensor<4x1x16x32xbf16>
  }
}
