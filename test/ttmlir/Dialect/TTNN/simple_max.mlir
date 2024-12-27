// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<512x32xbf16>) -> tensor<512xbf16> {
    %0 = tensor.empty() : tensor<512xbf16>
    // CHECK: %[[C:.*]] = "ttnn.max"[[C:.*]]
    %1 = "ttir.max"(%arg0, %0) <{dim = [1: i32], keep_dim = false}> : (tensor<512x32xbf16>, tensor<512xbf16>) -> tensor<512xbf16>
    return %1 : tensor<512xbf16>
  }
}
