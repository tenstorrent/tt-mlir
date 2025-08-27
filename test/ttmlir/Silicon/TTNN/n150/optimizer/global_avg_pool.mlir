// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

func.func @test_global_avg_pool(%arg0: tensor<1x32x112x112xf32>, %arg1: tensor<32x1x3x3xf32>, %arg2:  tensor<8x32x1x1xf32>) -> tensor<1x8x1x1xf32> {
    %0 = "ttir.constant"() <{value = dense<7.97193861E-5> : tensor<1x32xf32>}> : () -> tensor<1x32xf32>
    %1 = ttir.empty() : tensor<1x32x112x112xf32>
    // CHECK: "ttnn.conv2d"
    %2 = "ttir.convolution"(%arg0, %arg1, %1) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 32 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x32x112x112xf32>, tensor<32x1x3x3xf32>, tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
    // CHECK-NOT: "ttnn.reshape"
    // CHECK-NOT: "ttnn.permute"
    // CHECK: "ttnn.sum"
    %3 = ttir.empty() : tensor<1x32xf32>
    %4 = "ttir.sum"(%2, %3) <{dim_arg = [3 : i32, 2 : i32], keep_dim = false}> : (tensor<1x32x112x112xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
    %5 = ttir.empty() : tensor<1x32xf32>
    %6 = "ttir.multiply"(%4, %0, %5) : (tensor<1x32xf32>, tensor<1x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
    %7 = ttir.empty() : tensor<1x32x1x1xf32>
    %8 = "ttir.reshape"(%6, %7) <{shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    %10 = ttir.empty() : tensor<1x8x1x1xf32>
    // CHECK: "ttnn.conv2d"
    %11 = "ttir.convolution"(%8, %arg2, %10) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 0, 0, 0, 0>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 1, 1>}> : (tensor<1x32x1x1xf32>, tensor<8x32x1x1xf32>, tensor<1x8x1x1xf32>) -> tensor<1x8x1x1xf32>
    return %11 : tensor<1x8x1x1xf32>
}