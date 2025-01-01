// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @bitwise_xor_integer(%arg0: tensor<64x128xui16>) -> tensor<64x128xui16> {
    // CHECK-NOT: "ttir.bitwise_xor"
    // CHECK: "ttir.constant"
    // CHECK-SAME: value = dense<0> : tensor<64x128xui16>
    // CHECK-NOT: "ttir.bitwise_xor"
    %0 = tensor.empty() : tensor<64x128xui16>
    %1 = "ttir.bitwise_xor"(%arg0, %arg0, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xui16>, tensor<64x128xui16>, tensor<64x128xui16>) -> tensor<64x128xui16>
    return %1 : tensor<64x128xui16>
  }

  func.func @bitwise_xor_float(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-NOT: "ttir.bitwise_xor"
    // CHECK:  "ttir.constant"
    // CHECK-SAME: value = dense<0.000000e+00> : tensor<64x128xbf16>
    // CHECK-NOT: "ttir.bitwise_xor"
    %0 = tensor.empty() : tensor<64x128xbf16>
    %1 = "ttir.bitwise_xor"(%arg0, %arg0, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
