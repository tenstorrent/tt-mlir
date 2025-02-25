// RUN: ttmlir-opt %s | FileCheck %s
module attributes {} {
  func.func @index_1d(%arg0: tensor<64xbf16>) -> tensor<32xbf16> {
    %0 = tensor.empty() : tensor<32xbf16>
    // CHECK: = "ttir.index"
    %1 = "ttir.index"(%arg0, %0) <{dim = 0: i32, begin = 0: i32, end = 32: i32, step = 1: i32}> : (tensor<64xbf16>, tensor<32xbf16>) -> tensor<32xbf16>
    return %1 : tensor<32xbf16>
  }

  func.func @index_1d_step(%arg0: tensor<64xbf16>) -> tensor<16xbf16> {
    %0 = tensor.empty() : tensor<16xbf16>
    // CHECK: = "ttir.index"
    %1 = "ttir.index"(%arg0, %0) <{dim = 0: i32, begin = 0: i32, end = 32: i32, step = 2: i32}> : (tensor<64xbf16>, tensor<16xbf16>) -> tensor<16xbf16>
    return %1 : tensor<16xbf16>
  }

  func.func @index_2d(%arg0: tensor<128x64xbf16>) -> tensor<128x32xbf16> {
    %0 = tensor.empty() : tensor<128x32xbf16>
    // CHECK: = "ttir.index"
    %1 = "ttir.index"(%arg0, %0) <{dim = 1: i32, begin = 0: i32, end = 32: i32, step = 1: i32}> : (tensor<128x64xbf16>, tensor<128x32xbf16>) -> tensor<128x32xbf16>
    return %1 : tensor<128x32xbf16>
  }

  func.func @index_2d_step(%arg0: tensor<128x64xbf16>) -> tensor<128x16xbf16> {
    %0 = tensor.empty() : tensor<128x16xbf16>
    // CHECK: = "ttir.index"
    %1 = "ttir.index"(%arg0, %0) <{dim = 1: i32, begin = 32: i32, end = 64: i32, step = 2: i32}> : (tensor<128x64xbf16>, tensor<128x16xbf16>) -> tensor<128x16xbf16>
    return %1 : tensor<128x16xbf16>
  }

  func.func @index_3d(%arg0: tensor<3x128x64xbf16>) -> tensor<3x128x32xbf16> {
    %0 = tensor.empty() : tensor<3x128x32xbf16>
    // CHECK: = "ttir.index"
    %1 = "ttir.index"(%arg0, %0) <{dim = 2: i32, begin = 0: i32, end = 32: i32, step = 1: i32}> : (tensor<3x128x64xbf16>, tensor<3x128x32xbf16>) -> tensor<3x128x32xbf16>
    return %1 : tensor<3x128x32xbf16>
  }

  func.func @index_3d_step(%arg0: tensor<3x128x64xbf16>) -> tensor<3x128x8xbf16> {
    %0 = tensor.empty() : tensor<3x128x8xbf16>
    // CHECK: = "ttir.index"
    %1 = "ttir.index"(%arg0, %0) <{dim = 2: i32, begin = -1: i32, end = 0: i32, step = -8: i32}> : (tensor<3x128x64xbf16>, tensor<3x128x8xbf16>) -> tensor<3x128x8xbf16>
    return %1 : tensor<3x128x8xbf16>
  }

  func.func @index_4d(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x128x32xbf16> {
    %0 = tensor.empty() : tensor<10x3x128x32xbf16>
    // CHECK: = "ttir.index"
    %1 = "ttir.index"(%arg0, %0) <{dim = 3: i32, begin = 0: i32, end = 32: i32, step = 1: i32}> : (tensor<10x3x128x64xbf16>, tensor<10x3x128x32xbf16>) -> tensor<10x3x128x32xbf16>
    return %1 : tensor<10x3x128x32xbf16>
  }

  func.func @index_4d_step(%arg0: tensor<10x3x128x64xbf16>) -> tensor<10x3x128x24xbf16> {
    %0 = tensor.empty() : tensor<10x3x128x24xbf16>
    // CHECK: = "ttir.index"
    %1 = "ttir.index"(%arg0, %0) <{dim = 3: i32, begin = 0: i32, end = -16: i32, step = 2: i32}> : (tensor<10x3x128x64xbf16>, tensor<10x3x128x24xbf16>) -> tensor<10x3x128x24xbf16>
    return %1 : tensor<10x3x128x24xbf16>
  }
}
