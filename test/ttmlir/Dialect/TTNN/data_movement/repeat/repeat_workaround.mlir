// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK: %[[VAL0:[0-9]+]] = "ttnn.full"
    // CHECK-SAME: fillValue = 0.000000e+00 : f32
    // CHECK: %[[EMPTY:[0-9]+]] = "ttnn.empty"{{.*}}
    // CHECK: %{{[0-9]+}} = "ttnn.add"(%arg1, %[[VAL0]], %[[EMPTY]])
    // CHECK-NOT: "ttnn.repeat"
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %3 : tensor<1x16x32xf32>
  }
}
