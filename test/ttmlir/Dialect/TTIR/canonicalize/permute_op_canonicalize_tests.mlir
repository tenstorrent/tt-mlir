// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @permute_composition(%arg0: tensor<1x2x3x4x5xbf16>) -> tensor<3x2x1x5x4xbf16> {
    %0 = tensor.empty() : tensor<3x2x5x4x1xbf16>
    // CHECK: "ttir.permute"
    // CHECK-SAME: permutation = array<i64: 2, 1, 0, 4, 3>
    // CHECK-NOT: "ttir.permute"
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 2, 1, 4, 3, 0>}> : (tensor<1x2x3x4x5xbf16>, tensor<3x2x5x4x1xbf16>) -> tensor<3x2x5x4x1xbf16>
    %2 = tensor.empty() : tensor<3x2x1x5x4xbf16>
    %3 = "ttir.permute"(%1, %2) <{permutation = array<i64: 0, 1, 4, 2, 3>}> : (tensor<3x2x5x4x1xbf16>, tensor<3x2x1x5x4xbf16>) -> tensor<3x2x1x5x4xbf16>
    return %3 : tensor<3x2x1x5x4xbf16>
  }

  func.func @permute_noop(%arg0: tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16> {
    %0 = tensor.empty() : tensor<1x2x3x4x5xbf16>
    // CHECK-NOT: "ttir.permute"
    %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 0, 1, 2, 3, 4>}> : (tensor<1x2x3x4x5xbf16>, tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16>
    return %1 : tensor<1x2x3x4x5xbf16>
  }
}
