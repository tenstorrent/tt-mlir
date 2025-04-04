// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module @jit_eltwise_where {
  func.func public @test_where(%arg0: tensor<13x37xbf16>, %arg1: tensor<13x37xbf16>) -> tensor<13x37xbf16> {
    %0 = ttir.empty() : tensor<13x37xbf16>
    %1 = "ttir.eq"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
    %2 = ttir.empty() : tensor<13x37xbf16>
    %3 = "ttir.where"(%1, %arg0, %arg1, %2) <{operandSegmentSizes = array<i32: 3, 1>}> : (tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
    // CHECK: "ttnn.eq"
    // CHECK-SAME: tensor<13x37xbf16
    // CHECK-SAME: tensor<13x37xbf16
    // CHECK-SAME: -> tensor<13x37xbf16
    // CHECK: "ttnn.where"
    // CHECK-SAME: tensor<13x37xbf16
    // CHECK-SAME: tensor<13x37xbf16
    // CHECK-SAME: tensor<13x37xbf16
    // CHECK-SAME: -> tensor<13x37xbf16
     return %3 : tensor<13x37xbf16>
  }
}
