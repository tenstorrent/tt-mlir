// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_avgpool2d(%arg0: tensor<1x112x14x14xbf16>) -> tensor<1x112x7x7xbf16> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  %c = stablehlo.constant dense<4> : tensor<i64>
  // CHECK: %{{[0-9]+}} = "ttir.pooling"(%arg0, %{{[0-9]+}})
  // CHECK-SAME: base_dilations = array<i64: 1, 1, 1, 1>,
  // CHECK-SAME: padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>,
  // CHECK-SAME: pooling_method = #ttir<pooling_method Average>,
  // CHECK-SAME: window_dilations = array<i64: 1, 1, 1, 1>,
  // CHECK-SAME: window_dimensions = array<i64: 1, 1, 2, 2>,
  // CHECK-SAME: window_strides = array<i64: 1, 1, 2, 2>
  // CHECK-SAME: (tensor<1x112x14x14xbf16>, tensor<1x112x7x7xbf16>)
  // CHECK-SAME: -> tensor<1x112x7x7xbf16>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %5 = stablehlo.add %arg1, %arg2 : tensor<bf16>
    stablehlo.return %5 : tensor<bf16>
  }) : (tensor<1x112x14x14xbf16>, tensor<bf16>) -> tensor<1x112x7x7xbf16>
  %1 = stablehlo.convert %c : (tensor<i64>) -> tensor<bf16>
  %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x112x7x7xbf16>) -> tensor<1x112x7x7xbf16>
  %3 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<bf16>) -> tensor<1x112x7x7xbf16>
  %4 = stablehlo.divide %2, %3 : tensor<1x112x7x7xbf16>
  return %4 : tensor<1x112x7x7xbf16>
}
