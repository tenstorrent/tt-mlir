// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @linear_1d_1d(%arg0: tensor<3x64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<3x64x64xbf16> {
    %0 = tensor.empty() : tensor<3x64x64xbf16>
    // CHECK-NOT: "ttir.linear"
    // CHECK: "ttir.matmul"(%arg0, %arg1, %0)
    // CHECK-NOT: "ttir.linear"
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<3x64x128xbf16>, tensor<128x64xbf16>, tensor<3x64x64xbf16>) -> tensor<3x64x64xbf16>
    return %1 : tensor<3x64x64xbf16>
  }
}
