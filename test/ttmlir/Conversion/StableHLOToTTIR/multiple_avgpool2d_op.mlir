// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_multiple_avgpool2d(%arg0: tensor<1x32x128x128xbf16>) -> tensor<1x32x32x32xbf16> {
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%arg0)
  // CHECK-SAME: (tensor<1x32x128x128xbf16>) -> tensor<1x128x128x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.avg_pool2d"
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: -> tensor<1x64x64x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"
  // CHECK-SAME: (tensor<1x64x64x32xbf16>) -> tensor<1x32x64x64xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"
  // CHECK-SAME: (tensor<1x32x64x64xbf16>) -> tensor<1x64x64x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.avg_pool2d"
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: -> tensor<1x32x32x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"
  // CHECK-SAME: (tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"
  // CHECK-SAME: (tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.avg_pool2d"
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 1, 1>
  // CHECK-SAME: -> tensor<1x32x32x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"
  // CHECK-SAME: (tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16>
  // CHECK: return %{{[0-9]+}} : tensor<1x32x32x32xbf16>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  %c = stablehlo.constant dense<9> : tensor<i64>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %13 = stablehlo.add %arg1, %arg2 : tensor<bf16>
    stablehlo.return %13 : tensor<bf16>
  }) : (tensor<1x32x128x128xbf16>, tensor<bf16>) -> tensor<1x32x64x64xbf16>
  %1 = stablehlo.convert %c : (tensor<i64>) -> tensor<bf16>
  %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2, 3] : (tensor<1x32x64x64xbf16>) -> tensor<1x32x64x64xbf16>
  %3 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<bf16>) -> tensor<1x32x64x64xbf16>
  %4 = stablehlo.divide %2, %3 : tensor<1x32x64x64xbf16>
  %5 = "stablehlo.reduce_window"(%4, %cst) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %13 = stablehlo.add %arg1, %arg2 : tensor<bf16>
    stablehlo.return %13 : tensor<bf16>
  }) : (tensor<1x32x64x64xbf16>, tensor<bf16>) -> tensor<1x32x32x32xbf16>
  %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1, 2, 3] : (tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16>
  %7 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<bf16>) -> tensor<1x32x32x32xbf16>
  %8 = stablehlo.divide %6, %7 : tensor<1x32x32x32xbf16>
  %9 = "stablehlo.reduce_window"(%8, %cst) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 1, 1>}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %13 = stablehlo.add %arg1, %arg2 : tensor<bf16>
    stablehlo.return %13 : tensor<bf16>
  }) : (tensor<1x32x32x32xbf16>, tensor<bf16>) -> tensor<1x32x32x32xbf16>
  %10 = stablehlo.broadcast_in_dim %9, dims = [0, 1, 2, 3] : (tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16>
  %11 = stablehlo.divide %10, %7 : tensor<1x32x32x32xbf16>
  return %11 : tensor<1x32x32x32xbf16>
}
