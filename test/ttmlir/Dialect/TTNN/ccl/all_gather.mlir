// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x128xbf16> {
    %0 = tensor.empty() : tensor<1x1x32x128xbf16>
    %1 = "ttir.all_gather"(%arg0, %0) <{dim = 3 : si32}> : (tensor<1x1x32x32xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.all_gather"[[C:.*]]
    return %1 : tensor<1x1x32x128xbf16>
  }
}
