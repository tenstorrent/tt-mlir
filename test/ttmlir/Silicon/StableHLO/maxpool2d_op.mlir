// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s
// UNSUPPORTED: true

func.func public @test_maxpool2d(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x64x64x32xbf16> {
  %0 = stablehlo.constant dense<0xFF80> : tensor<bf16>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<bf16>) -> tensor<bf16>
  %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>, window_dimensions = array<i64: 1, 3, 3, 1>, window_strides = array<i64: 1, 2, 2, 1>}> ({
  ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
    %3 = stablehlo.maximum %arg2, %arg3 : tensor<bf16>
    stablehlo.return %3 : tensor<bf16>
  }) : (tensor<1x128x128x32xbf16>, tensor<bf16>) -> tensor<1x64x64x32xbf16>
  return %2 : tensor<1x64x64x32xbf16>
}
