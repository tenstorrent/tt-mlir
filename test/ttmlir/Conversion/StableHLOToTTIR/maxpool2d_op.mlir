// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file -stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func public @test_maxpool2d_nhwc(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%arg0)
  // CHECK-SAME: (tensor<1x128x128x32xbf16>)
  // CHECK-SAME: -> tensor<1x64x64x32xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x128x128x32xbf16>, tensor<bf16>) -> tensor<1x64x64x32xbf16>
  return %2 : tensor<1x64x64x32xbf16>
}

// -----

func.func public @test_maxpool2d_nchw_with_permute(%arg0: tensor<1x32x128x128xbf16>) -> tensor<1x32x64x64xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%arg0)
  // CHECK-SAME: (tensor<1x32x128x128xbf16>) -> tensor<1x128x128x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x64x64x32xbf16>) -> tensor<1x32x64x64xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x32x128x128xbf16>, tensor<bf16>) -> tensor<1x32x64x64xbf16>
  return %2 : tensor<1x32x64x64xbf16>
}

// -----

func.func public @test_maxpool2d_with_reshape(%arg0: tensor<128x128xbf16>) -> tensor<64x64xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%arg0)
  // CHECK-SAME: (tensor<128x128xbf16>) -> tensor<1x128x128x1xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x128x128x1xbf16>) -> tensor<1x64x64x1xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x64x64x1xbf16>) -> tensor<64x64xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 2, 2>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<128x128xbf16>, tensor<bf16>) -> tensor<64x64xbf16>
  return %2 : tensor<64x64xbf16>
}
