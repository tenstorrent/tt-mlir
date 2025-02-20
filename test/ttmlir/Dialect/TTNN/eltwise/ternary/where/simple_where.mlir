// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module @jit_eltwise_where {
  func.func public @test_where(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xf32>) -> tensor<13x37xf32> {
    %0 = tensor.empty() : tensor<13x37xf32>
    %1 = "ttir.eq"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    %2 = tensor.empty() : tensor<13x37xf32>
    %3 = "ttir.where"(%1, %arg0, %arg1, %2) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    // CHECK: %[[VAL1:[0-9]+]] = "ttnn.eq"(%arg0, %arg1)
    // CHECK: %{{[0-9]+}} = "ttnn.where"(%[[VAL1]], %arg0, %arg1)
     return %3 : tensor<13x37xf32>
  }
}
