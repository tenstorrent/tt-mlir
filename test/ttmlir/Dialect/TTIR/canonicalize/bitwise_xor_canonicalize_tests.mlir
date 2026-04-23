// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @bitwise_xor_integer(%arg0: tensor<64x128xui16>) -> tensor<64x128xui16> {
    // CHECK-NOT: "ttir.bitwise_xor"
    // CHECK: "ttir.zeros"
    // CHECK-SAME: tensor<64x128xui16>
    // CHECK-NOT: "ttir.bitwise_xor"
    %1 = "ttir.bitwise_xor"(%arg0, %arg0) : (tensor<64x128xui16>, tensor<64x128xui16>) -> tensor<64x128xui16>
    return %1 : tensor<64x128xui16>
  }

  func.func @bitwise_xor_float(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-NOT: "ttir.bitwise_xor"
    // CHECK:  "ttir.zeros"
    // CHECK-SAME: tensor<64x128xbf16>
    // CHECK-NOT: "ttir.bitwise_xor"
    %1 = "ttir.bitwise_xor"(%arg0, %arg0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
