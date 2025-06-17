// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @permute_reshape_permute_to_reshape(%arg0: tensor<16x7x7x2048xbf16>) -> tensor<16x1x49x2048xbf16> {
    // CHECK: "ttir.reshape"
    // CHECK: shape = [16 : i32, 1 : i32, 49 : i32, 2048 : i32]
    %0 = ttir.empty() : tensor<16x2048x7x7xbf16>
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<16x7x7x2048xbf16>, tensor<16x2048x7x7xbf16>) -> tensor<16x2048x7x7xbf16>
    %2 = ttir.empty() : tensor<16x1x2048x49xbf16>
    %3 = "ttir.reshape"(%1, %2) <{shape = [16 : i32, 1 : i32, 2048 : i32, 49 : i32]}> : (tensor<16x2048x7x7xbf16>, tensor<16x1x2048x49xbf16>) -> tensor<16x1x2048x49xbf16>
    %4 = ttir.empty() : tensor<16x1x49x2048xbf16>
    %5 = "ttir.permute"(%3, %4) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<16x1x2048x49xbf16>, tensor<16x1x49x2048xbf16>) -> tensor<16x1x49x2048xbf16>
    return %5 : tensor<16x1x49x2048xbf16>
  }
}
