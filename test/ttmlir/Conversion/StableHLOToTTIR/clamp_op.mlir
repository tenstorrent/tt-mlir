// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_transpose attributes {} {
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
