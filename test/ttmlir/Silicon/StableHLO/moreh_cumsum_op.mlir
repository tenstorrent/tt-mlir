// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s \
// RUN:     --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @moreh_cumsum attributes {} {
  func.func @test_moreh_cumsum_dim0(%arg0: tensor<8x2x4x16xbf16>) -> tensor<8x2x4x16xbf16> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_dim0
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 0 : i64
    // CHECK-SAME: tensor<8x2x4x16xbf16,
    // CHECK-SAME: tensor<8x2x4x16xbf16,
    // CHECK-SAME: -> tensor<8x2x4x16xbf16,
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[7, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 8, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<8x2x4x16xbf16>, tensor<bf16>) -> tensor<8x2x4x16xbf16>
    return %0 : tensor<8x2x4x16xbf16>
  }

  func.func @test_moreh_cumsum_dim1(%arg0: tensor<8x2x4x16xbf16>) -> tensor<8x2x4x16xbf16> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_dim1
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 1 : i64
    // CHECK-SAME: tensor<8x2x4x16xbf16,
    // CHECK-SAME: tensor<8x2x4x16xbf16,
    // CHECK-SAME: -> tensor<8x2x4x16xbf16,
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [1, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 2, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<8x2x4x16xbf16>, tensor<bf16>) -> tensor<8x2x4x16xbf16>
    return %0 : tensor<8x2x4x16xbf16>
  }

  func.func @test_moreh_cumsum_dim2(%arg0: tensor<8x2x4x16xbf16>) -> tensor<8x2x4x16xbf16> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_dim2
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 2 : i64
    // CHECK-SAME: tensor<8x2x4x16xbf16,
    // CHECK-SAME: tensor<8x2x4x16xbf16,
    // CHECK-SAME: -> tensor<8x2x4x16xbf16,
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [0, 0], [3, 0], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 4, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<8x2x4x16xbf16>, tensor<bf16>) -> tensor<8x2x4x16xbf16>
    return %0 : tensor<8x2x4x16xbf16>
  }

  func.func @test_moreh_cumsum_dim3(%arg0: tensor<8x2x4x16xbf16>) -> tensor<8x2x4x16xbf16> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_dim3
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 3 : i64
    // CHECK-SAME: tensor<8x2x4x16xbf16,
    // CHECK-SAME: tensor<8x2x4x16xbf16,
    // CHECK-SAME: -> tensor<8x2x4x16xbf16,
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<[[0, 0], [0, 0], [0, 0], [15, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 16>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<8x2x4x16xbf16>, tensor<bf16>) -> tensor<8x2x4x16xbf16>
    return %0 : tensor<8x2x4x16xbf16>
  }
}
