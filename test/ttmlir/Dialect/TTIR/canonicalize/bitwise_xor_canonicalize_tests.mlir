// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @bitwise_xor_integer(%arg0: tensor<64x128xui16>) -> tensor<64x128xui16> {
    // CHECK-NOT: "ttir.bitwise_xor"
    // CHECK: "ttir.full"
    // CHECK-SAME: fill_value = 0 : i32
    // CHECK-SAME: tensor<64x128xui16>
    // CHECK-NOT: "ttir.bitwise_xor"
    %0 = ttir.empty() : tensor<64x128xui16>
    %1 = "ttir.bitwise_xor"(%arg0, %arg0, %0) : (tensor<64x128xui16>, tensor<64x128xui16>, tensor<64x128xui16>) -> tensor<64x128xui16>
    return %1 : tensor<64x128xui16>
  }

  func.func @bitwise_xor_float(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-NOT: "ttir.bitwise_xor"
    // CHECK:  "ttir.full"
    // CHECK-SAME: fill_value = 0.000000e+00 : f32
    // CHECK-SAME: tensor<64x128xbf16>
    // CHECK-NOT: "ttir.bitwise_xor"
    %0 = ttir.empty() : tensor<64x128xbf16>
    %1 = "ttir.bitwise_xor"(%arg0, %arg0, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}
