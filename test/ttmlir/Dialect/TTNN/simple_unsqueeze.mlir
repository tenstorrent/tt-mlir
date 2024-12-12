// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x1x2x32x32xbf16> {
    %0 = tensor.empty() : tensor<4x1x2x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = -4 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x1x2x32x32xbf16>) -> tensor<4x1x2x32x32xbf16>
    return %1 : tensor<4x1x2x32x32xbf16>
  }
}
