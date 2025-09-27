// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for matmul operation
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: error: 'ttir.arange' op Output tensor shape must be 16 at dim 1 (since start=0, end=32, step=2), but got 32
    %1 = "ttir.arange"() <{start = 0: si64, dtype = i64, end = 32: si64, step = 2: si64, arange_dimension = 1: i64}> : () -> tensor<1x32x128x128xf32>
    %dps = ttir.empty() : tensor<1x32x128x128xf32>
    %2 = "ttir.multiply"(%arg0, %1, %dps) : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %2 : tensor<1x32x128x128xf32>
  }
}
