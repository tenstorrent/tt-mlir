// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s| FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x2x1x32x32xbf16>) -> tensor<1x2x32x32xbf16> {
    %0 = tensor.empty() : tensor<1x2x32x32xbf16>
    // CHECK: = "ttnn.reshape"
    %1 = "ttir.squeeze"(%arg0, %0) <{dim = -3 : si32}> : (tensor<1x2x1x32x32xbf16>, tensor<1x2x32x32xbf16>) -> tensor<1x2x32x32xbf16>
    return %1 : tensor<1x2x32x32xbf16>
  }
}
