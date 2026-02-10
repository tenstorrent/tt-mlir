// REQUIRES: stablehlo
// RUN: ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func public @test_maxpool2d_nhwc(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%arg0)
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x128x128x32xbf16>, tensor<bf16>) -> tensor<1x64x64x32xbf16>
  return %2 : tensor<1x64x64x32xbf16>
}

// -----

func.func public @test_maxpool2d_nchw_with_permute(%arg0: tensor<1x32x128x128xbf16>) -> tensor<1x32x64x64xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%arg0)
  // CHECK-SAME: (tensor<1x32x128x128xbf16>) -> tensor<1x128x128x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x64x64x32xbf16>) -> tensor<1x32x64x64xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x32x128x128xbf16>, tensor<bf16>) -> tensor<1x32x64x64xbf16>
  return %2 : tensor<1x32x64x64xbf16>
}

// -----

func.func public @test_maxpool2d_with_reshape(%arg0: tensor<128x128xbf16>) -> tensor<64x64xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%arg0)
  // CHECK-SAME: (tensor<128x128xbf16>) -> tensor<1x128x128x1xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x128x128x1xbf16>) -> tensor<1x64x64x1xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x64x64x1xbf16>) -> tensor<64x64xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 3>, window_strides = array<i64: 2, 2>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<128x128xbf16>, tensor<bf16>) -> tensor<64x64xbf16>
  return %2 : tensor<64x64xbf16>
}

// -----

func.func public @test_maxpool2d_with_pad_fusion(%arg0: tensor<1x112x112x64xbf16>) -> tensor<1x58x58x64xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %pad_value = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  // CHECK-NOT: "ttir.pad"
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%arg0)
  // CHECK-SAME: padding = array<i32: 3, 3, 3, 3>
  // CHECK-SAME: (tensor<1x112x112x64xbf16>) -> tensor<1x58x58x64xbf16>
  %padded = "stablehlo.pad"(%arg0, %pad_value) {
    edge_padding_low = array<i64: 0, 1, 1, 0>,
    edge_padding_high = array<i64: 0, 1, 1, 0>,
    interior_padding = array<i64: 0, 0, 0, 0>
  } : (tensor<1x112x112x64xbf16>, tensor<bf16>) -> tensor<1x114x114x64xbf16>
  %result = "stablehlo.reduce_window"(%padded, %0) <{
    padding = dense<[[0, 0], [2, 2], [2, 2], [0, 0]]> : tensor<4x2xi64>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>
  }> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %max = stablehlo.maximum %arg1, %arg2 : tensor<bf16>
    stablehlo.return %max : tensor<bf16>
  }) : (tensor<1x114x114x64xbf16>, tensor<bf16>) -> tensor<1x58x58x64xbf16>
  return %result : tensor<1x58x58x64xbf16>
}

// -----

func.func public @test_maxpool2d_nchw_with_pad_fusion(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x58x58xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %pad_value = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  // CHECK-NOT: "ttir.pad"
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%arg0)
  // CHECK-SAME: (tensor<1x64x112x112xbf16>) -> tensor<1x112x112x64xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: padding = array<i32: 3, 3, 3, 3>
  // CHECK-SAME: (tensor<1x112x112x64xbf16>) -> tensor<1x58x58x64xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x58x58x64xbf16>) -> tensor<1x64x58x58xbf16>
  %padded = "stablehlo.pad"(%arg0, %pad_value) {
    edge_padding_low = array<i64: 0, 0, 1, 1>,
    edge_padding_high = array<i64: 0, 0, 1, 1>,
    interior_padding = array<i64: 0, 0, 0, 0>
  } : (tensor<1x64x112x112xbf16>, tensor<bf16>) -> tensor<1x64x114x114xbf16>
  %result = "stablehlo.reduce_window"(%padded, %0) <{
    padding = dense<[[0, 0], [0, 0], [2, 2], [2, 2]]> : tensor<4x2xi64>,
    window_dimensions = array<i64: 1, 1, 3, 3>,
    window_strides = array<i64: 1, 1, 2, 2>
  }> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %max = stablehlo.maximum %arg1, %arg2 : tensor<bf16>
    stablehlo.return %max : tensor<bf16>
  }) : (tensor<1x64x114x114xbf16>, tensor<bf16>) -> tensor<1x64x58x58xbf16>
  return %result : tensor<1x64x58x58xbf16>
}

// -----

func.func public @test_maxpool2d_pad_no_fusion_batch_pad(%arg0: tensor<1x112x112x64xbf16>) -> tensor<2x55x55x64xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %pad_value = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  // CHECK: "ttir.pad"
  // CHECK: "ttir.max_pool2d"
  %padded = "stablehlo.pad"(%arg0, %pad_value) {
    edge_padding_low = array<i64: 1, 0, 0, 0>,
    edge_padding_high = array<i64: 0, 0, 0, 0>,
    interior_padding = array<i64: 0, 0, 0, 0>
  } : (tensor<1x112x112x64xbf16>, tensor<bf16>) -> tensor<2x112x112x64xbf16>
  %result = "stablehlo.reduce_window"(%padded, %0) <{
    padding = dense<0> : tensor<4x2xi64>,
    window_dimensions = array<i64: 1, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 1>
  }> ({
  ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
    %max = stablehlo.maximum %arg1, %arg2 : tensor<bf16>
    stablehlo.return %max : tensor<bf16>
  }) : (tensor<2x112x112x64xbf16>, tensor<bf16>) -> tensor<2x55x55x64xbf16>
  return %result : tensor<2x55x55x64xbf16>
}
