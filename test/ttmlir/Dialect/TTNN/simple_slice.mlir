// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<4x32x32xbf16>) -> tensor<2x16x16xbf16> {
    %0 = tensor.empty() : tensor<2x16x16xbf16>
    // CHECK: %[[C:.*]] = "ttnn.slice"[[C:.*]]
    %1 = "ttir.slice"(%arg0, %0) <{begins = [0: i32, 0: i32, 0: i32], ends = [2: i32, 16: i32, 16: i32], step = [1: i32, 1: i32, 1: i32]}> : (tensor<4x32x32xbf16>, tensor<2x16x16xbf16>) -> tensor<2x16x16xbf16>
    return %1 : tensor<2x16x16xbf16>
  }
}
