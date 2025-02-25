// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x32xbf16>, %arg1: tensor<512x128xbf16>) -> tensor<1x32x128xbf16> {
    // CHECK: = "ttnn.empty"
    %0 = tensor.empty() : tensor<1x32x128xbf16>
    // CHECK: = "ttnn.embedding"
    %1 = "ttir.embedding"(%arg0, %arg1, %0) : (tensor<1x32xbf16>, tensor<512x128xbf16>, tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16>
    return %1 : tensor<1x32x128xbf16>
  }
}
