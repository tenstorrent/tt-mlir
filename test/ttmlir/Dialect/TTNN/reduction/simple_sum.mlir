// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
    %0 = tensor.empty() : tensor<512x1xbf16>
    // CHECK: %[[C:.*]] = "ttnn.sum"[[C:.*]]
    %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<512x1024xbf16>, tensor<512x1xbf16>) -> tensor<512x1xbf16>
    return %1 : tensor<512x1xbf16>
  }
}
