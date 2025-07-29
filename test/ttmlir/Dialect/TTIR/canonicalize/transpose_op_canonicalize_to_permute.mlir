// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @transpose_involution(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK-NOT: "ttir.transpose"
    // CHECK-NOT: "ttir.permute"
    %0 = ttir.empty() : tensor<128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
    %2 = ttir.empty() : tensor<64x128xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = 1 : si32, dim1 = 0 : si32}> : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %3 : tensor<64x128xbf16>
  }

  func.func @transpose_normalize_range(%arg0: tensor<32x64x128xbf16>) -> tensor<32x128x64xbf16> {
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 1>
    %0 = ttir.empty() : tensor<32x128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -1 : si32, dim1 = -2 : si32}> : (tensor<32x64x128xbf16>, tensor<32x128x64xbf16>) -> tensor<32x128x64xbf16>
    return %1 : tensor<32x128x64xbf16>
  }
}
