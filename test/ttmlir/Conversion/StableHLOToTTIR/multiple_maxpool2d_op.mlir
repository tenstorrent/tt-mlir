// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func public @test_maxpool2d(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x32x32x32xbf16> {
  // CHECK: %[[POOL1:[0-9]+]] = "ttir.max_pool2d"(%arg0)
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16>
  // CHECK: %[[POOL2:[0-9]+]] = "ttir.max_pool2d"(%[[POOL1]])
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 2, 2>
  // CHECK-SAME: (tensor<1x64x64x32xbf16>) -> tensor<1x32x32x32xbf16>
  // CHECK: %[[POOL3:[0-9]+]] = "ttir.max_pool2d"(%[[POOL2]])
  // CHECK-SAME: kernel = array<i32: 3, 3>
  // CHECK-SAME: stride = array<i32: 1, 1>
  // CHECK-SAME: (tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16>
  // CHECK: return %[[POOL3]]
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x128x128x32xbf16>, tensor<bf16>) -> tensor<1x64x64x32xbf16>
  %4 = "stablehlo.reduce_window"(%2, %0) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg4: tensor<bf16>, %arg5: tensor<bf16>):
    %5 = stablehlo.maximum %arg4, %arg5 : tensor<bf16>
    stablehlo.return %5 : tensor<bf16>
  }) : (tensor<1x64x64x32xbf16>, tensor<bf16>) -> tensor<1x32x32x32xbf16>
  %6 = "stablehlo.reduce_window"(%4, %0) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
  ^bb0(%arg6: tensor<bf16>, %arg7: tensor<bf16>):
    %7 = stablehlo.maximum %arg6, %arg7 : tensor<bf16>
    stablehlo.return %7 : tensor<bf16>
  }) : (tensor<1x32x32x32xbf16>, tensor<bf16>) -> tensor<1x32x32x32xbf16>
  return %6 : tensor<1x32x32x32xbf16>
}
