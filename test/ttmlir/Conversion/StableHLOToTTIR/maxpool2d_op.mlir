// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
func.func public @test_maxpool2d(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<bf16>
  %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x128x128x32xbf16>, tensor<bf16>) -> tensor<1x64x64x32xbf16>
  // CHECK: = tensor.empty
  // CHECK: = "ttir.pooling"
  return %2 : tensor<1x64x64x32xbf16>
}
