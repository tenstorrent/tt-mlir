// REQUIRES: stablehlo
// RUN: ttmlir-opt --split-input-file --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// 5D NCDHW max_pool3d decomposed into two max_pool2d passes (channel-preserving).
// [N,C,D,H,W] -> permute -> [N,D,H,W,C] -> reshape [N*D,H,W,C] -> MaxPool2d HW ->
// reshape [N,D,Hout*Wout,C] -> MaxPool2d [kD,1] -> permute -> [N,C,Dout,flat] ->
// reshape [N,C,Dout,Hout,Wout].
// Input: [1, 2, 4, 8, 8], kernel: [3, 3, 3], stride: [2, 2, 2], padding: [1, 1, 1]
func.func public @test_maxpool3d_ncdhw(%arg0: tensor<1x2x4x8x8xbf16>) -> tensor<1x2x2x4x4xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%arg0)
  // CHECK-SAME: (tensor<1x2x4x8x8xbf16>) -> tensor<1x4x8x8x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x4x8x8x2xbf16>) -> tensor<4x8x8x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 3>, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<4x8x8x2xbf16>) -> tensor<4x4x4x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<4x4x4x2xbf16>) -> tensor<1x4x16x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 1>, padding = array<i32: 1, 0, 1, 0>, stride = array<i32: 2, 1>
  // CHECK-SAME: (tensor<1x4x16x2xbf16>) -> tensor<1x2x16x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x2x16x2xbf16>) -> tensor<1x2x2x16xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x2x2x16xbf16>) -> tensor<1x2x2x4x4xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{
    padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]> : tensor<5x2xi64>,
    window_dimensions = array<i64: 1, 1, 3, 3, 3>,
    window_strides = array<i64: 1, 1, 2, 2, 2>
  }> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x2x4x8x8xbf16>, tensor<bf16>) -> tensor<1x2x2x4x4xbf16>
  return %2 : tensor<1x2x2x4x4xbf16>
}

// -----

// 5D NDHWC max_pool3d: permute to NCDHW, then same decomposition, permute back.
// Input: [1, 4, 8, 8, 2] (NDHWC), kernel: [3, 3, 3], stride: [2, 2, 2], padding: [1, 1, 1]
func.func public @test_maxpool3d_ndhwc(%arg0: tensor<1x4x8x8x2xbf16>) -> tensor<1x2x4x4x2xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%arg0)
  // CHECK-SAME: (tensor<1x4x8x8x2xbf16>) -> tensor<1x2x4x8x8xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x2x4x8x8xbf16>) -> tensor<1x4x8x8x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x4x8x8x2xbf16>) -> tensor<4x8x8x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 3>, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<4x8x8x2xbf16>) -> tensor<4x4x4x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<4x4x4x2xbf16>) -> tensor<1x4x16x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 1>, padding = array<i32: 1, 0, 1, 0>, stride = array<i32: 2, 1>
  // CHECK-SAME: (tensor<1x4x16x2xbf16>) -> tensor<1x2x16x2xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x2x16x2xbf16>) -> tensor<1x2x2x16xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x2x2x16xbf16>) -> tensor<1x2x2x4x4xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x2x2x4x4xbf16>) -> tensor<1x2x4x4x2xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{
    padding = dense<[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]> : tensor<5x2xi64>,
    window_dimensions = array<i64: 1, 3, 3, 3, 1>,
    window_strides = array<i64: 1, 2, 2, 2, 1>
  }> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x4x8x8x2xbf16>, tensor<bf16>) -> tensor<1x2x4x4x2xbf16>
  return %2 : tensor<1x2x4x4x2xbf16>
}

// -----

// Matches the exact configuration from issue #7240: Dense UNet 3D max_pool3d.
// Input: [1, 96, 16, 128, 128], kernel: [3, 3, 3], stride: [2, 2, 2],
// padding: [1, 1, 1], dilation: [1, 1, 1]
func.func public @test_maxpool3d_dense_unet(%arg0: tensor<1x96x16x128x128xbf16>) -> tensor<1x96x8x64x64xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%arg0)
  // CHECK-SAME: (tensor<1x96x16x128x128xbf16>) -> tensor<1x16x128x128x96xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x16x128x128x96xbf16>) -> tensor<16x128x128x96xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 3>, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<16x128x128x96xbf16>) -> tensor<16x64x64x96xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<16x64x64x96xbf16>) -> tensor<1x16x4096x96xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.max_pool2d"(%{{[0-9]+}})
  // CHECK-SAME: ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 3, 1>, padding = array<i32: 1, 0, 1, 0>, stride = array<i32: 2, 1>
  // CHECK-SAME: (tensor<1x16x4096x96xbf16>) -> tensor<1x8x4096x96xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.permute"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x8x4096x96xbf16>) -> tensor<1x96x8x4096xbf16>
  // CHECK: %{{[0-9]+}} = "ttir.reshape"(%{{[0-9]+}})
  // CHECK-SAME: (tensor<1x96x8x4096xbf16>) -> tensor<1x96x8x64x64xbf16>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{
    padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]> : tensor<5x2xi64>,
    window_dimensions = array<i64: 1, 1, 3, 3, 3>,
    window_strides = array<i64: 1, 1, 2, 2, 2>
  }> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x96x16x128x128xbf16>, tensor<bf16>) -> tensor<1x96x8x64x64xbf16>
  return %2 : tensor<1x96x8x64x64xbf16>
}
