// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @reverse_composition(%arg0: tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16> {
    // CHECK: "ttir.reverse"
    // CHECK-SAME: dimensions = array<i64: 1, 4>
    // CHECK-NOT: "ttir.reverse"
    %0 = tensor.empty() : tensor<1x2x3x4x5xbf16>
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 1, 0, 3>}> : (tensor<1x2x3x4x5xbf16>, tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16>
    %2 = tensor.empty() : tensor<1x2x3x4x5xbf16>
    %3 = "ttir.reverse"(%1, %2) <{dimensions = array<i64: 0, 3, 4>}> : (tensor<1x2x3x4x5xbf16>, tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16>
    return %3 : tensor<1x2x3x4x5xbf16>
  }

  func.func @reverse_noop(%arg0: tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16> {
    // CHECK-NOT: "ttir.reverse"
    %0 = tensor.empty() : tensor<1x2x3x4x5xbf16>
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64>}> : (tensor<1x2x3x4x5xbf16>, tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16>
    return %1 : tensor<1x2x3x4x5xbf16>
  }


    func.func @reverse_composition_noop(%arg0: tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16> {
    // CHECK-NOT: "ttir.reverse"
    %0 = tensor.empty() : tensor<1x2x3x4x5xbf16>
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 1, 0, 3>}> : (tensor<1x2x3x4x5xbf16>, tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16>
    %2 = tensor.empty() : tensor<1x2x3x4x5xbf16>
    %3 = "ttir.reverse"(%1, %2) <{dimensions = array<i64: 3, 1, 0>}> : (tensor<1x2x3x4x5xbf16>, tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16>
    return %3 : tensor<1x2x3x4x5xbf16>
  }
}
