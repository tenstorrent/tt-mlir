// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: %[[ARANGE:[0-9]+]] = "ttir.arange"
    // CHECK-SAME: {arange_dimension = 3 : i64, end = 32 : si64, start = 0 : si64, step = 1 : si64}
    // CHECK-SAME: -> tensor<1x1x1x32xf32>
    // CHECK: %[[TRANSPOSE:[0-9]+]] = "ttir.transpose"(%[[ARANGE]],
    // CHECK-SAME: {dim0 = 1 : si32, dim1 = 3 : si32}>
    // CHECK-SAME: (tensor<1x1x1x32xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%[[TRANSPOSE]],
    // CHECK-SAME: broadcast_dimensions = array<i32: 1, 1, 128, 128>
    // CHECK-SAME: (tensor<1x32x1x1xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %1 = "ttir.arange"() <{start = 0: si64, end = 32: si64, step = 1: si64, arange_dimension = 1: i64}> : () -> tensor<1x32x128x128xf32>
    %dps = tensor.empty() : tensor<1x32x128x128xf32>
    %2 = "ttir.multiply"(%arg0, %1, %dps) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %2 : tensor<1x32x128x128xf32>
  }
}
