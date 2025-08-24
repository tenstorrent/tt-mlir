// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xbf16>, %arg1: tensor<512x128xbf16>) -> tensor<32x32x128xbf16> {
    %0 = ttir.empty() : tensor<32x32x128xbf16>
    // CHECK: = "ttnn.embedding"
    %1 = "ttir.embedding"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<512x128xbf16>, tensor<32x32x128xbf16>) -> tensor<32x32x128xbf16>
    return %1 : tensor<32x32x128xbf16>
  }
}
