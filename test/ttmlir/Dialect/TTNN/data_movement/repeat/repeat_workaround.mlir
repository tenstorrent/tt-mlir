// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x1x32xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<1x16x32xf32> {
    // CHECK: %[[VAL0:[0-9]+]] = "ttnn.full"
    // CHECK-SAME: fillValue = 0.000000e+00 : f32
    // CHECK: %{{[0-9]+}} = "ttnn.add"(%arg1, %{{[0-9]+}})
    // CHECK-NOT: "ttnn.repeat"
    %0 = ttir.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = ttir.empty() : tensor<1x16x32xf32>
    %3 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 1, 16, 32>}> : (tensor<1x1x1xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %4 = ttir.empty() : tensor<1x16x32xf32>
    %5 = "ttir.multiply"(%1, %3, %4) : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %5 : tensor<1x16x32xf32>
  }
}
