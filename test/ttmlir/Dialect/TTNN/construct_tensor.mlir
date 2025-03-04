// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
    // CHECK: "ttnn.construct_tensor"
    %0 = tensor.empty() : tensor<2x4x32x32xbf16>
    return %0 : tensor<2x4x32x32xbf16>
  }
}
