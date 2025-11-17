// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_test_select_and_scatter attributes {} {
  func.func public @test_select_and_scatter(%arg0: tensor<1x1x4x4xbf16>) -> tensor<1x1x4x4xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1x1x2x2xbf16>
    // CHECK: "ttir.permute"
    // CHECK: <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x1x4x4xbf16>, tensor<1x4x4x1xbf16>) -> tensor<1x4x4x1xbf16>
    // CHECK: "ttir.max_pool2d_with_indices"
    // CHECK: ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 2, 2>, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 2, 2>
    // CHECK: "ttir.reshape"
    // CHECK: <{shape = [1 : i32, 4 : i32, 1 : i32, 1 : i32]}> : (tensor<1x2x2x1xi32>, tensor<1x4x1x1xi32>) -> tensor<1x4x1x1xi32>
    // CHECK: "ttir.scatter_in_dim"
    // CHECK: "ttir.reshape"
    // CHECK: <{shape = [1 : i32, 4 : i32, 4 : i32, 1 : i32]}> : (tensor<1x16x1x1xbf16>, tensor<1x4x4x1xbf16>) -> tensor<1x4x4x1xbf16>
    // CHECK: "ttir.permute"
    // CHECK: <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x4x4x1xbf16>, tensor<1x1x4x4xbf16>) -> tensor<1x1x4x4xbf16>
    %0 = "stablehlo.select_and_scatter"(%arg0, %cst_0, %cst) <{window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>}> ({
      ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
        %1 = stablehlo.compare GE, %arg1, %arg2 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
        stablehlo.return %1 : tensor<i1>
      }, {
      ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
        %1 = stablehlo.add %arg1, %arg2 : tensor<bf16>
        stablehlo.return %1 : tensor<bf16>
      }) : (tensor<1x1x4x4xbf16>, tensor<1x1x2x2xbf16>, tensor<bf16>) -> tensor<1x1x4x4xbf16>
    return %0 : tensor<1x1x4x4xbf16>
  }

  func.func public @test_select_and_scatter_padding(%arg0: tensor<1x1x4x4xbf16>) -> tensor<1x1x4x4xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<1x1x3x3xbf16>
    // CHECK: "ttir.permute"
    // CHECK: <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x1x4x4xbf16>, tensor<1x4x4x1xbf16>) -> tensor<1x4x4x1xbf16>
    // CHECK: "ttir.max_pool2d_with_indices"
    // CHECK: ceil_mode = false, dilation = array<i32: 1, 1>, kernel = array<i32: 2, 2>, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>
    // CHECK: "ttir.reshape"
    // CHECK: <{shape = [1 : i32, 9 : i32, 1 : i32, 1 : i32]}> : (tensor<1x3x3x1xi32>, tensor<1x9x1x1xi32>) -> tensor<1x9x1x1xi32>
    // CHECK: "ttir.scatter_in_dim"
    // CHECK: "ttir.reshape"
    // CHECK: <{shape = [1 : i32, 4 : i32, 4 : i32, 1 : i32]}> : (tensor<1x16x1x1xbf16>, tensor<1x4x4x1xbf16>) -> tensor<1x4x4x1xbf16>
    // CHECK: "ttir.permute"
    // CHECK: <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x4x4x1xbf16>, tensor<1x1x4x4xbf16>) -> tensor<1x1x4x4xbf16>
    %0 = "stablehlo.select_and_scatter"(%arg0, %cst_0, %cst)
        <{
            window_dimensions = array<i64: 1, 1, 2, 2>,
            window_strides = array<i64: 1, 1, 2, 2>,
            padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>
          }> ({
      ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
        %1 = stablehlo.compare GE, %arg1, %arg2 : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
        stablehlo.return %1 : tensor<i1>
      }, {
      ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
        %1 = stablehlo.add %arg1, %arg2 : tensor<bf16>
        stablehlo.return %1 : tensor<bf16>
      }) : (tensor<1x1x4x4xbf16>, tensor<1x1x3x3xbf16>, tensor<bf16>) -> tensor<1x1x4x4xbf16>
    return %0 : tensor<1x1x4x4xbf16>
  }
}
