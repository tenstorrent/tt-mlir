// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @broadcast_noop(%arg0: tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16> {
    // CHECK-NOT: "ttir.broadcast"
    %0 = tensor.empty() : tensor<1x2x3x4x5xbf16>
    %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 1, 1, 1, 1>}> : (tensor<1x2x3x4x5xbf16>, tensor<1x2x3x4x5xbf16>) -> tensor<1x2x3x4x5xbf16>
    return %1 : tensor<1x2x3x4x5xbf16>
  }
}
