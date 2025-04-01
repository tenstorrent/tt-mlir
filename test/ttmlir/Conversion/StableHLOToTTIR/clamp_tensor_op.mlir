// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @test_clamp_tensor attributes {} {
  func.func public @test_clamp_tensor(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-LABEL: func.func public @test_clamp_tensor
    // CHECK: %[[EMPTY0:.*]] = ttir.empty() : [[TENSOR:tensor<4xf32>]]
    // CHECK: %[[CLAMP:.*]] = "ttir.clamp_tensor"(%arg0, %arg1, %arg2, %[[EMPTY0]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]], [[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.clamp %arg1, %arg0, %arg2 : tensor<4xf32>
    // CHECK: return %[[CLAMP]] : [[TENSOR]]
    return %0 : tensor<4xf32>
  }

  func.func public @test_clamp_tensor_constant(%arg0: tensor<1x16xbf16>, %arg1: tensor<bf16>) -> tensor<1x16xbf16> {
    // CHECK-LABEL: func.func public @test_clamp_tensor_constant(
    // CHECK: %[[CONSTANT:[0-9]+]] = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %cst = arith.constant dense<3.0> : tensor<1xf64>
    // CHECK: %[[CAST:[0-9]+]] = "ttir.typecast"(%[[CONSTANT]],
    // CHECK-SAME: (tensor<1xf32>, tensor<1xbf16>) -> tensor<1xbf16>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xbf16>
    // CHECK: %[[RESHAPE0:[0-9]+]] = "ttir.reshape"(%[[CAST]],
    // CHECK-SAME: shape = [1 : i32]
    // CHECK-SAME: (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"(%[[RESHAPE0]],
    // CHECK-SAME: shape = [1 : i32, 1 : i32]
    // CHECK-SAME: (tensor<1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    // CHECK: %[[MIN:[0-9]+]] = "ttir.broadcast"(%[[RESHAPE1]]
    // CHECK-SAME: {broadcast_dimensions = array<i64: 1, 16>}
    // CHECK-SAME:  (tensor<1x1xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%arg1,
    // CHECK-SAME: <{shape = [1 : i32, 1 : i32]}>
    // CHECK-SAME:  (tensor<1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    // CHECK: %[[MAX:[0-9]+]] = "ttir.broadcast"(%[[RESHAPE2]],
    // CHECK-SAME: <{broadcast_dimensions = array<i64: 1, 16>}>
    // CHECK-SAME: (tensor<1x1xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    // CHECK: = "ttir.clamp_tensor"(%arg0, %[[MIN]], %[[MAX]],
    // CHECK-SAME: (tensor<1x16xbf16>, tensor<1x16xbf16>, tensor<1x16xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    %2 = stablehlo.clamp %1, %arg0, %arg1 : (tensor<bf16>, tensor<1x16xbf16>, tensor<bf16>) -> tensor<1x16xbf16>
    return %2 : tensor<1x16xbf16>
  }
}
