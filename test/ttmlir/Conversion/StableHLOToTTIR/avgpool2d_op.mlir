// REQUIRES: stablehlo
// RUN: ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @test_avgpool2d_nhwc(%arg0: tensor<1x14x14x112xbf16>) -> tensor<1x7x7x112xbf16> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.avg_pool2d"(%arg0)
  // CHECK-SAME: kernel = array<i32: 2, 2>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x14x14x112xbf16>) -> tensor<1x7x7x112xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.constant"
  // CHECK-SAME: value = dense<4.00
  // CHECK: %{{[0-9]+}} = "ttir.multiply"(%{{[0-9]+}}, %{{[0-9]+}})
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %5 = stablehlo.add %arg1, %arg2 : tensor<bf16>
    stablehlo.return %5 : tensor<bf16>
  }) : (tensor<1x14x14x112xbf16>, tensor<bf16>) -> tensor<1x7x7x112xbf16>
  return %0 : tensor<1x7x7x112xbf16>
}

// -----

func.func @test_avgpool2d_nchw_with_permute(%arg0: tensor<1x112x14x14xbf16>) -> tensor<1x112x7x7xbf16> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%arg0)
  // CHECK-SAME: (tensor<1x112x14x14xbf16>) -> tensor<1x14x14x112xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.avg_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: kernel = array<i32: 2, 2>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x14x14x112xbf16>) -> tensor<1x7x7x112xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.constant"
  // CHECK-SAME: value = dense<4.00
  // CHECK: %{{[0-9]+}} = "ttir.multiply"(%{{[0-9]+}}, %{{[0-9]+}})
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x7x7x112xbf16>) -> tensor<1x112x7x7xbf16>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %5 = stablehlo.add %arg1, %arg2 : tensor<bf16>
    stablehlo.return %5 : tensor<bf16>
  }) : (tensor<1x112x14x14xbf16>, tensor<bf16>) -> tensor<1x112x7x7xbf16>
  return %0 : tensor<1x112x7x7xbf16>
}

// -----

func.func public @test_avgpool2d_with_reshape(%arg0: tensor<128x128xbf16>) -> tensor<64x64xbf16> {
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%arg0)
  // CHECK-SAME: (tensor<128x128xbf16>) -> tensor<1x128x128x1xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.avg_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x128x128x1xbf16>) -> tensor<1x64x64x1xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x64x64x1xbf16>) -> tensor<64x64xbf16>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 2, 2>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.add %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<128x128xbf16>, tensor<bf16>) -> tensor<64x64xbf16>
  return %0 : tensor<64x64xbf16>
}
