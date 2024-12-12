// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
    %0 = tensor.empty() : tensor<2x4x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 32: i32]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
    return %1 : tensor<2x4x32x32xbf16>
  }
}
