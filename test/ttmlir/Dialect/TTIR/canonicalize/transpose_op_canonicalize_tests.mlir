// RUN: ttmlir-opt -canonicalize %s | FileCheck %s

module {
  func.func @transpose_to_permute(%arg0: tensor<2x4x64x128xbf16>) -> tensor<4x64x2x128xbf16> {
    // CHECK-LABEL: func.func @transpose_to_permute
    // CHECK-NOT: "ttir.transpose"
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 1, 2, 0, 3>
    // CHECK-NOT: "ttir.transpose"
    %0 = ttir.empty() : tensor<4x2x64x128xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<2x4x64x128xbf16>, tensor<4x2x64x128xbf16>) -> tensor<4x2x64x128xbf16>
    %2 = ttir.empty() : tensor<4x64x2x128xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<4x2x64x128xbf16>, tensor<4x64x2x128xbf16>) -> tensor<4x64x2x128xbf16>
    return %3 : tensor<4x64x2x128xbf16>
  }

  func.func @transpose_to_permute_all_negative_dims(%arg0: tensor<2x4x64x128xbf16>) -> tensor<2x128x4x64xbf16> {
    // CHECK-LABEL: func.func @transpose_to_permute_all_negative_dims
    // CHECK-NOT: "ttir.transpose"
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-NOT: "ttir.transpose"
    %0 = ttir.empty() : tensor<2x4x128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<2x4x64x128xbf16>, tensor<2x4x128x64xbf16>) -> tensor<2x4x128x64xbf16>
    %2 = ttir.empty() : tensor<2x128x4x64xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<2x4x128x64xbf16>, tensor<2x128x4x64xbf16>) -> tensor<2x128x4x64xbf16>
    return %3 : tensor<2x128x4x64xbf16>
  }

  func.func @chained_transpose_to_permute(%arg0: tensor<2x4x64x128xbf16>) -> tensor<4x128x2x64xbf16> {
    // CHECK-LABEL: func.func @chained_transpose_to_permute
    // CHECK-NOT: "ttir.transpose"
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 1, 3, 0, 2>
    // CHECK-NOT: "ttir.transpose"
    %0 = ttir.empty() : tensor<2x4x128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 2 : si32, dim1 = -1 : si32}> : (tensor<2x4x64x128xbf16>, tensor<2x4x128x64xbf16>) -> tensor<2x4x128x64xbf16>
    %2 = ttir.empty() : tensor<4x2x128x64xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = 1 : si32, dim1 = -4 : si32}> : (tensor<2x4x128x64xbf16>, tensor<4x2x128x64xbf16>) -> tensor<4x2x128x64xbf16>
    %4 = ttir.empty() : tensor<4x128x2x64xbf16>
    %5 = "ttir.transpose"(%3, %4) <{dim0 = -3 : si32, dim1 = -2 : si32}> : (tensor<4x2x128x64xbf16>, tensor<4x128x2x64xbf16>) -> tensor<4x128x2x64xbf16>
    return %5 : tensor<4x128x2x64xbf16>
  }

  func.func @transpose_involution(%arg0: tensor<2x4x64x128xbf16>) -> tensor<2x4x64x128xbf16> {
    // CHECK-LABEL: func.func @transpose_involution
    // CHECK-SAME: %[[INPUT:.*]]:
    // CHECK-NOT: "ttir.transpose"
    // CHECK-NOT: "ttir.permute"
    // CHECK: return %[[INPUT]]
    %0 = ttir.empty() : tensor<2x4x128x64xbf16>
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<2x4x64x128xbf16>, tensor<2x4x128x64xbf16>) -> tensor<2x4x128x64xbf16>
    %2 = ttir.empty() : tensor<2x4x64x128xbf16>
    %3 = "ttir.transpose"(%1, %2) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<2x4x128x64xbf16>, tensor<2x4x64x128xbf16>) -> tensor<2x4x64x128xbf16>
    return %3 : tensor<2x4x64x128xbf16>
  }
}
