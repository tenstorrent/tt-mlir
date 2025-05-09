// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

func.func public @test_sumpool2d(%arg0: tensor<8x256x6x6xf32>) -> tensor<8x256x6x6xf32> {
  // CHECK-LABEL: @test_sumpool2d
  %c = stablehlo.constant dense<1> : tensor<i32>
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[POOL:[0-9]+]] = "ttir.pooling"(%arg0, %{{[0-9]+}})
  // CHECK-SAME: base_dilations = array<i64: 1, 1, 1, 1>
  // CHECK-SAME: padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>,
  // CHECK-SAME: pooling_method = #ttir<pooling_method Average>,
  // CHECK-SAME: window_dilations = array<i64: 1, 1, 1, 1>,
  // CHECK-SAME: window_dimensions = array<i64: 1, 1, 1, 1>,
  // CHECK-SAME: window_strides = array<i64: 1, 1, 1, 1>}
  // CHECK-SAME: (tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>)
  // CHECK-SAME: -> tensor<8x256x6x6xf32>
  // CHECK: %[[CONSTANT:[0-9]+]] = "ttir.constant"()
  // CHECK-SAME: value = dense<1.000000e+00>
  // CHECK-SAME: -> tensor<8x256x6x6xf32>
  // CHECK: %{{[0-9]+}} = "ttir.multiply"(%[[POOL]], %[[CONSTANT]], %{{[0-9]+}})
  // CHECK-SAME: (tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>)
  // CHECK-SAME: -> tensor<8x256x6x6xf32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst_0) <{window_dimensions = array<i64: 1, 1, 1, 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %6 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %6 : tensor<f32>
  }) : (tensor<8x256x6x6xf32>, tensor<f32>) -> tensor<8x256x6x6xf32>
  %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x256x6x6xf32>
  %2 = stablehlo.convert %c : (tensor<i32>) -> tensor<f32>
  %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<8x256x6x6xf32>
  %4 = stablehlo.multiply %1, %3 : tensor<8x256x6x6xf32>
  %5 = stablehlo.divide %0, %4 : tensor<8x256x6x6xf32>
  return %5 : tensor<8x256x6x6xf32>
}
