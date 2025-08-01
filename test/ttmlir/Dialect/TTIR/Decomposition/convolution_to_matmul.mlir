// RUN: ttmlir-opt --ttir-conv2d-to-matmul %s | FileCheck %s

module {
  func.func @test_conv_1x1_to_matmul(%arg0: tensor<1x7x7x512xf32>, %arg1: tensor<2048x512x1x1xf32>) -> tensor<1x7x7x2048xf32> {
    // CHECK-LABEL: func.func @test_conv_1x1_to_matmul
    // CHECK: %[[RESHAPE_INPUT:.*]] = "ttir.reshape"(%arg0, %[[SHAPE_INPUT:.*]])
    // CHECK: %[[PERMUTE_FILTER:.*]] = "ttir.permute"(%arg1, %[[PERM_SHAPE:.*]])
    // CHECK: %[[RESHAPE_FILTER:.*]] = "ttir.reshape"(%[[PERMUTE_FILTER]], %[[SHAPE_FILTER:.*]])
    // CHECK: %[[MATMUL:.*]] = "ttir.matmul"(%[[RESHAPE_INPUT]], %[[RESHAPE_FILTER]], %[[MATMUL_OUT:.*]])
    // CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[MATMUL]], %[[FINAL_SHAPE:.*]])
    // CHECK: return %[[RESHAPE_OUT]]
    %0 = ttir.empty() : tensor<1x7x7x2048xf32>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x7x7x512xf32>, tensor<2048x512x1x1xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    return %1 : tensor<1x7x7x2048xf32>
  }
  func.func @test_conv_1x1_with_bias_to_linear(%arg0: tensor<1x7x7x512xf32>, %arg1: tensor<2048x512x1x1xf32>, %arg2: tensor<1x1x1x2048xf32>) -> tensor<1x7x7x2048xf32> {
    // CHECK-LABEL: func.func @test_conv_1x1_with_bias_to_linear
    // CHECK: %[[RESHAPE_INPUT:.*]] = "ttir.reshape"(%arg0, %[[SHAPE_INPUT:.*]])
    // CHECK: %[[PERMUTE_FILTER:.*]] = "ttir.permute"(%arg1, %[[PERM_SHAPE:.*]])
    // CHECK: %[[RESHAPE_FILTER:.*]] = "ttir.reshape"(%[[PERMUTE_FILTER]], %[[SHAPE_FILTER:.*]])
    // CHECK: %[[RESHAPE_BIAS:.*]] = "ttir.reshape"(%arg2, %[[SHAPE_BIAS:.*]])
    // CHECK: %[[LINEAR:.*]] = "ttir.linear"(%[[RESHAPE_INPUT]], %[[RESHAPE_FILTER]], %[[RESHAPE_BIAS]], %[[LINEAR_OUT:.*]])
    // CHECK: %[[RESHAPE_OUT:.*]] = "ttir.reshape"(%[[LINEAR]], %[[FINAL_SHAPE:.*]])
    // CHECK: return %[[RESHAPE_OUT]]
    %0 = ttir.empty() : tensor<1x7x7x2048xf32>
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x7x7x512xf32>, tensor<2048x512x1x1xf32>, tensor<1x1x1x2048xf32>, tensor<1x7x7x2048xf32>) -> tensor<1x7x7x2048xf32>
    return %1 : tensor<1x7x7x2048xf32>
  }
}
