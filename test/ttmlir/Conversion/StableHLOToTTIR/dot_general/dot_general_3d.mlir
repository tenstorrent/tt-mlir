// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<8x1x920xbf16>, %arg1: tensor<8x100x32xbf16>, %arg2: tensor<8x32x920xbf16>) -> tensor<8x100x920xbf16> {
    %0 = stablehlo.broadcast_in_dim %arg2, dims = [0, 1, 2] : (tensor<8x32x920xbf16>) -> tensor<8x32x920xbf16>
    // CHECK: %[[C:.*]] = "ttir.matmul"[[C:.*]]
    %1 = stablehlo.dot_general %arg1, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<8x100x32xbf16>, tensor<8x32x920xbf16>) -> tensor<8x100x920xbf16>
    return %1 : tensor<8x100x920xbf16>
  }
}
