// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
// UNSUPPORTED: true
// https://github.com/tenstorrent/tt-mlir/issues/1448
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: %[[C:.*]] = "ttnn.arange"[[C:.*]]
    %1 = "ttir.arange"() <{start = 0: si64, end = 32: si64, step = 1: si64, arange_dimension = 1: i64}> : () -> tensor<1x32x128x128xf32>
    %dps = tensor.empty() : tensor<1x32x128x128xf32>
    %2 = "ttir.multiply"(%arg0, %1, %dps) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %2 : tensor<1x32x128x128xf32>
  }
}
