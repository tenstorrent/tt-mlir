// REQUIRES: stablehlo
// RUN: not ttmlir-opt --stablehlo-to-ttir-pipeline %s 2>&1 | FileCheck %s

// Negative test: 5D avg_pool3d (stablehlo.add + zero init) should be rejected.
// The 5D decomposition only supports max pooling because the associative
// decomposition max(D,H,W) = max_D(max(H,W)) does not hold for average pooling.
// CHECK: error: failed to legalize operation 'stablehlo.reduce_window'
func.func public @test_avgpool3d_rejected(%arg0: tensor<1x2x4x8x8xbf16>) -> tensor<1x2x2x4x4xbf16> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
  %2 = "stablehlo.reduce_window"(%arg0, %0) <{
    padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]> : tensor<5x2xi64>,
    window_dimensions = array<i64: 1, 1, 3, 3, 3>,
    window_strides = array<i64: 1, 1, 2, 2, 2>
  }> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.add %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x2x4x8x8xbf16>, tensor<bf16>) -> tensor<1x2x2x4x4xbf16>
  return %2 : tensor<1x2x2x4x4xbf16>
}
