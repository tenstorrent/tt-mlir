// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x128xbf16> {
    %0 = tensor.empty() : tensor<1x1x32x128xbf16>
    %1 = "ttir.all_gather"(%arg0, %0) <{all_gather_dim = 3 : si32, channel_handle = 1 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<1x1x32x32xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.all_gather"[[C:.*]]
    return %1 : tensor<1x1x32x128xbf16>
  }
}
