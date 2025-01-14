// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @matmul_canonicalize_lhs(%arg0: tensor<64x128xbf16>) -> tensor<128x128xbf16> {
    %0 = tensor.empty() : tensor<128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
    %2 = tensor.empty() : tensor<128x128xbf16>
    // CHECK-NOT: "ttir.transpose"
    // CHECK: "ttir.matmul"
    // CHECK-SAME: transpose_a = true
    // CHECK-SAME: transpose_b = false
    %3 = "ttir.matmul"(%1, %arg0, %2) : (tensor<128x64xbf16>, tensor<64x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %3 : tensor<128x128xbf16>
  }
  func.func @matmul_canonicalize_rhs(%arg0: tensor<64x128xbf16>) -> tensor<64x64xbf16> {
    %0 = tensor.empty() : tensor<128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
    %2 = tensor.empty() : tensor<64x64xbf16>
    // CHECK-NOT: "ttir.transpose"
    // CHECK: "ttir.matmul"
    // CHECK-SAME: transpose_a = false
    // CHECK-SAME: transpose_b = true
    %3 = "ttir.matmul"(%arg0, %1, %2) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
  func.func @matmul_double_transpose_lhs(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64x64xbf16> {
    %0 = tensor.empty() : tensor<128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
    %2 = tensor.empty() : tensor<64x64xbf16>
    // CHECK-NOT: "ttir.transpose"
    // CHECK: "ttir.matmul"
    // CHECK-SAME: transpose_a = false
    // CHECK-SAME: transpose_b = false
    %3 = "ttir.matmul"(%1, %arg1, %2) <{transpose_a = true}> : (tensor<128x64xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %3 : tensor<64x64xbf16>
  }
}
