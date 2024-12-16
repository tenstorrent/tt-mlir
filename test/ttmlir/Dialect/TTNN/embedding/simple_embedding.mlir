// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xbf16>, %arg1: tensor<512x128xbf16>) -> tensor<32x32x128xbf16> {
    %0 = tensor.empty() : tensor<32x32x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.embedding"[[C:.*]]
    %1 = "ttir.embedding"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<512x128xbf16>, tensor<32x32x128xbf16>) -> tensor<32x32x128xbf16>
    return %1 : tensor<32x32x128xbf16>
  }
}
