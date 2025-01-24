// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x32xbf16> {
    %0 = tensor.empty() : tensor<512x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.mean"[[C:.*]]
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<512x1024xbf16>, tensor<512x32xbf16>) -> tensor<512x32xbf16>
    return %1 : tensor<512x32xbf16>
  }
}
