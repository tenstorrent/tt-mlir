// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_clamp attributes {} {
  func.func public @test_clamp_constant(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
    // CHECK: %[[EMPTY:.*]] = tensor.empty() : [[TENSOR:tensor<4xf32>]]
    // CHECK: "ttir.clamp"(%arg0, %[[EMPTY]])
    // CHECK-SAME: max = 3.000000e+00 : f32, min = 2.000000e+00 : f32
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.clamp %cst, %arg0, %cst_0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @test_clamp_constant_1(%arg0: tensor<1x16xbf16>) -> tensor<1x16xbf16> {
    // CHECK-LABEL: @test_clamp_constant_1(
    %cst = arith.constant dense<3.0> : tensor<1xf64>
    %cst_0 = arith.constant dense<6> : tensor<1xi64>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    %2 = stablehlo.convert %cst_0 : (tensor<1xi64>) -> tensor<1xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1xbf16>) -> tensor<bf16>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : [[TENSOR:tensor<1x16xbf16>]]
    // CHECK: "ttir.clamp"(%arg0, %[[EMPTY]])
    // CHECK-SAME: max = 6.000000e+00 : f32, min = 3.000000e+00 : f32
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %4 = stablehlo.clamp %1, %arg0, %3 : (tensor<bf16>, tensor<1x16xbf16>, tensor<bf16>) -> tensor<1x16xbf16>
    return %4 : tensor<1x16xbf16>
  }

  func.func public @test_clamp_constant_2(%arg0: tensor<1x32xbf16>) -> (tensor<1x32xbf16>) {
    // CHECK-LABEL: @test_clamp_constant_2(
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<5.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x32xbf16>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<1x32xbf16>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : [[TENSOR:tensor<1x32xbf16>]]
    // CHECK: "ttir.clamp"(%arg0, %[[EMPTY]])
    // CHECK-SAME: max = 5.000000e+00 : f32, min = 2.000000e+00 : f32
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %2 = stablehlo.clamp %0, %arg0, %1 : tensor<1x32xbf16>
    return %2 : tensor<1x32xbf16>
  }

  func.func public @test_clamp_tensor(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: %[[EMPTY0:.*]] = tensor.empty() : [[TENSOR:tensor<4xf32>]]
    // CHECK: %[[MAX:.*]] = "ttir.maximum"(%arg1, %arg0, %[[EMPTY0]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    // CHECK: %[[EMPTY1:.*]] = tensor.empty() : [[TENSOR]]
    // CHECK: %[[MIN:.*]] = "ttir.minimum"(%[[MAX]], %arg2, %[[EMPTY1]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.clamp %arg1, %arg0, %arg2 : tensor<4xf32>
    // CHECK: return %[[MIN]] : [[TENSOR]]
    return %0 : tensor<4xf32>
  }
}
