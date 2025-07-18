// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

func.func public @test_sumpool2d(%arg0: tensor<1x1x4x8xf32>) -> (tensor<1x1x2x4xf32>) {
  // CHECK-LABEL: @test_sumpool2d
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[POOL:[0-9]+]] = "ttir.pooling"(%arg0, %{{[0-9]+}})
  // CHECK-SAME: base_dilations = array<i64: 1, 1, 1, 1>
  // CHECK-SAME: padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>,
  // CHECK-SAME: pooling_method = #ttir<pooling_method Sum>,
  // CHECK-SAME: window_dilations = array<i64: 1, 1, 1, 1>,
  // CHECK-SAME: window_dimensions = array<i64: 1, 1, 2, 2>,
  // CHECK-SAME: window_strides = array<i64: 1, 1, 2, 2>}
  // CHECK-SAME: (tensor<1x1x4x8xf32>, tensor<1x1x2x4xf32>)
  // CHECK-SAME: -> tensor<1x1x2x4xf32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) <{base_dilations = array<i64: 1, 1, 1, 1>, padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) : (tensor<1x1x4x8xf32>, tensor<f32>) -> tensor<1x1x2x4xf32>
  return %1 : tensor<1x1x2x4xf32>
}

func.func @test_complex_sum2d(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>) ->
                      (tensor<2x2xf32>, tensor<2x2xi32>) {
  // CHECK-LABEL: @test_complex_sum2d
  %init0 = stablehlo.constant dense<0.0> : tensor<f32>
  %init1 = stablehlo.constant dense<0> : tensor<i32>
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %init0, %init1) ({
         ^bb0(%a0: tensor<f32>, %a1: tensor<i32>,
                %b0: tensor<f32>, %b1: tensor<i32>):
              %2 = stablehlo.add %a0, %b0 : tensor<f32>
              %3 = stablehlo.add %a1, %b1 : tensor<i32>
              "stablehlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
            })
         { padding = dense<[[2, 2], [0, 0]]> : tensor<2x2xi64>,
           window_dimensions = array<i64: 5, 1>,
           window_strides = array<i64: 3, 1>}
         : (tensor<4x2xf32>, tensor<4x2xi32>, tensor<f32>, tensor<i32>) ->
              (tensor<2x2xf32>, tensor<2x2xi32>)
  func.return %0#0, %0#1 : tensor<2x2xf32>, tensor<2x2xi32>
}

func.func public @test_avgpool2d_workaround(%arg0: tensor<8x256x6x6xf32>) -> tensor<8x256x6x6xf32> {
  // CHECK-LABEL: @test_avgpool2d_workaround
  %c = stablehlo.constant dense<1> : tensor<i32>
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %{{[0-9]+}} = "ttir.pooling"(%arg0, %{{[0-9]+}})
  // CHECK-SAME: base_dilations = array<i64: 1, 1, 1, 1>
  // CHECK-SAME: padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>,
  // CHECK-SAME: pooling_method = #ttir<pooling_method Sum>,
  // CHECK-SAME: window_dilations = array<i64: 1, 1, 1, 1>,
  // CHECK-SAME: window_dimensions = array<i64: 1, 1, 1, 1>,
  // CHECK-SAME: window_strides = array<i64: 1, 1, 1, 1>}
  // CHECK-SAME: (tensor<8x256x6x6xf32>, tensor<8x256x6x6xf32>)
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
