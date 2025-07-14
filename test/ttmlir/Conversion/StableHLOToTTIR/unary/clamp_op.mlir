// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_clamp attributes {} {
  func.func public @test_clamp_constant(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-LABEL: func.func public @test_clamp_constant
    // CHECK: %[[MIN:[0-9]+]] = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<4xf32>
    // CHECK: %[[MAX:[0-9]+]] = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    %cst_0 = stablehlo.constant dense<3.000000e+00> : tensor<4xf32>
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : [[TENSOR:tensor<4xf32>]]
    // CHECK: "ttir.clamp_tensor"(%arg0, %[[MIN]], %[[MAX]], %[[EMPTY]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]], [[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.clamp %cst, %arg0, %cst_0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func public @test_clamp_indirect_constant_reshape(%arg0: tensor<1x16xbf16>) -> tensor<1x16xbf16> {
    // CHECK-LABEL: func.func public @test_clamp_indirect_constant_reshape
    %cst = arith.constant dense<3.0> : tensor<1xf64>
    %cst_0 = arith.constant dense<6> : tensor<1xi64>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xbf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    %2 = stablehlo.convert %cst_0 : (tensor<1xi64>) -> tensor<1xbf16>
    %3 = stablehlo.reshape %2 : (tensor<1xbf16>) -> tensor<bf16>
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<1x16xbf16>
    // CHECK: %{{[0-9]+}} = "ttir.clamp_tensor"(%arg0, %{{[0-9]+}}, %{{[0-9]+}}, %[[EMPTY]])
    // CHECK-SAME: : (tensor<1x16xbf16>, tensor<bf16>, tensor<bf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    %4 = stablehlo.clamp %1, %arg0, %3 : (tensor<bf16>, tensor<1x16xbf16>, tensor<bf16>) -> tensor<1x16xbf16>
    return %4 : tensor<1x16xbf16>
  }

  func.func public @test_clamp_indirect_constant_broadcast(%arg0: tensor<1x32xbf16>) -> (tensor<1x32xbf16>) {
    // CHECK-LABEL: func.func public @test_clamp_indirect_constant_broadcast
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<5.000000e+00> : tensor<bf16>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<1x32xbf16>
    %1 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<bf16>) -> tensor<1x32xbf16>
    // CHECK: "ttir.clamp_tensor"(%arg0, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: (tensor<1x32xbf16>, tensor<1x32xbf16>, tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %2 = stablehlo.clamp %0, %arg0, %1 : tensor<1x32xbf16>
    return %2 : tensor<1x32xbf16>
  }

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
    // CHECK: %[[CONSTANT:[0-9]+]] = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<1xf64>}> : () -> tensor<1xf64>
    %cst = arith.constant dense<3.0> : tensor<1xf64>
    // CHECK: %[[CAST:[0-9]+]] = "ttir.typecast"(%[[CONSTANT]],
    // CHECK-SAME: (tensor<1xf64>, tensor<1xbf16>) -> tensor<1xbf16>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xbf16>
    // CHECK: %[[MIN:[0-9]+]] = "ttir.reshape"(%[[CAST]],
    // CHECK-SAME: shape = []
    // CHECK-SAME: (tensor<1xbf16>, tensor<bf16>) -> tensor<bf16>
    %1 = stablehlo.reshape %0 : (tensor<1xbf16>) -> tensor<bf16>
    // CHECK: = "ttir.clamp_tensor"(%arg0, %[[MIN]], %arg1, %{{[0-9]+}})
    // CHECK-SAME: (tensor<1x16xbf16>, tensor<bf16>, tensor<bf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    %2 = stablehlo.clamp %1, %arg0, %arg1 : (tensor<bf16>, tensor<1x16xbf16>, tensor<bf16>) -> tensor<1x16xbf16>
    return %2 : tensor<1x16xbf16>
  }
}
