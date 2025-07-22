// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @moreh_cumsum attributes {} {
  func.func @test_moreh_cumsum_0_dim1(%arg0: tensor<1x10xi64>) -> tensor<1x10xi64> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_0_dim1
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<1x10xi64>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.cumsum"(%arg0, %[[EMPTY]])
    // CHECK-SAME: <{dim = 1 : i64}>
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = "stablehlo.reduce_window"(%arg0, %c) <{padding = dense<[[0, 0], [9, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 1>, window_dimensions = array<i64: 1, 10>, window_strides = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i64>
      stablehlo.return %1 : tensor<i64>
    }) : (tensor<1x10xi64>, tensor<i64>) -> tensor<1x10xi64>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<1x10xi64>
  }

  func.func @test_moreh_cumsum_1_dim0(%arg0: tensor<5xi64>) -> tensor<5xi64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    // CHECK-LABEL: func.func @test_moreh_cumsum_1_dim0
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<5xi64>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.cumsum"(%arg0, %[[EMPTY]])
    // CHECK-SAME: <{dim = 0 : i64}>
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = "stablehlo.reduce_window"(%arg0, %c) <{padding = dense<[[4, 0]]> : tensor<1x2xi64>, window_dilations = array<i64: 1>, window_dimensions = array<i64: 5>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i64>
      stablehlo.return %1 : tensor<i64>
    }) : (tensor<5xi64>, tensor<i64>) -> tensor<5xi64>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<5xi64>
  }

  func.func @test_moreh_cumsum_2_dim0(%arg0: tensor<8x2x4x16xi32>) -> tensor<8x2x4x16xi64> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_2_dim0
    // CHECK: ttir.empty() : [[TENSOR:tensor<8x2x4x16xi64>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.cumsum"
    // CHECK-SAME: <{dim = 0 : i64}>
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.convert %arg0 : (tensor<8x2x4x16xi32>) -> tensor<8x2x4x16xi64>
    %1 = "stablehlo.reduce_window"(%0, %c) <{padding = dense<[[7, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 8, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i64>
      stablehlo.return %2 : tensor<i64>
    }) : (tensor<8x2x4x16xi64>, tensor<i64>) -> tensor<8x2x4x16xi64>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %1 : tensor<8x2x4x16xi64>
  }

  func.func @test_moreh_cumsum_2_dim2(%arg0: tensor<8x2x4x16xi32>) -> tensor<8x2x4x16xi64> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_2_dim2
    // CHECK: ttir.empty() : [[TENSOR:tensor<8x2x4x16xi64>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.cumsum"
    // CHECK-SAME: <{dim = 2 : i64}>
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.convert %arg0 : (tensor<8x2x4x16xi32>) -> tensor<8x2x4x16xi64>
    %1 = "stablehlo.reduce_window"(%0, %c) <{padding = dense<[[0, 0], [0, 0], [3, 0], [0, 0]]> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 4, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i64>
      stablehlo.return %2 : tensor<i64>
    }) : (tensor<8x2x4x16xi64>, tensor<i64>) -> tensor<8x2x4x16xi64>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %1 : tensor<8x2x4x16xi64>
  }

  func.func @test_moreh_cumsum_3_dim1(%arg0: tensor<8x1x4x16xi32>) -> tensor<8x1x4x16xi64> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_3_dim1
    // CHECK: ttir.empty() : [[TENSOR:tensor<8x1x4x16xi64>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.cumsum"
    // CHECK-SAME: <{dim = 1 : i64}>
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.convert %arg0 : (tensor<8x1x4x16xi32>) -> tensor<8x1x4x16xi64>
    %1 = "stablehlo.reduce_window"(%0, %c) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i64>
      stablehlo.return %2 : tensor<i64>
    }) : (tensor<8x1x4x16xi64>, tensor<i64>) -> tensor<8x1x4x16xi64>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %1 : tensor<8x1x4x16xi64>
  }

  func.func @test_moreh_cumsum_3_dim3(%arg0: tensor<8x2x4x1xbf16>) -> tensor<8x2x4x1xbf16> {
    // CHECK-LABEL: func.func @test_moreh_cumsum_3_dim3
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<8x2x4x1xbf16>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.cumsum"(%arg0, %[[EMPTY]])
    // CHECK-SAME: <{dim = 3 : i64}>
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %0 = "stablehlo.reduce_window"(%arg0, %cst) <{padding = dense<0> : tensor<4x2xi64>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 1, 1>, window_strides = array<i64: 1, 1, 1, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<bf16>
      stablehlo.return %1 : tensor<bf16>
    }) : (tensor<8x2x4x1xbf16>, tensor<bf16>) -> tensor<8x2x4x1xbf16>
    // CHECK: return %[[RET]] : [[TENSOR]]
    return %0 : tensor<8x2x4x1xbf16>
  }

  func.func @test_no_padding(%arg0: tensor<1x1xi32>) -> tensor<1x1xi32> {
    // CHECK-LABEL: func.func @test_no_padding
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : [[TENSOR:tensor<1x1xi32>]]
    // CHECK: %[[RET:[0-9]+]] = "ttir.cumsum"(%arg0, %[[EMPTY]])
    // CHECK-SAME: <{dim = 0 : i64}>
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<i32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{window_dimensions = array<i64: 1, 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }) : (tensor<1x1xi32>, tensor<i32>) -> tensor<1x1xi32>
    return %1 : tensor<1x1xi32>
  }
}
