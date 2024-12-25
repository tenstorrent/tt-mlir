// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module {
  func.func @simple_linear_without_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<64x64xbf16
    %0 = tensor.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.matmul"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }

  func.func @simple_linear_with_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
    // CHECK: "ttnn.empty"
    // CHECK-SAME: tensor<64x64xbf16
    %0 = tensor.empty() : tensor<64x64xbf16>
    // CHECK: "ttnn.linear"
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<128x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    // CHECK-SAME: tensor<64x64xbf16
    %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
}
