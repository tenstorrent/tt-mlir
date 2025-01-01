// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x256x512xf32>, %arg1: tensor<1024x256x1xf32>, %arg2: tensor<1024xf32>) -> tensor<1x1024x512xf32> {
    %0 = tensor.empty() : tensor<1x1024x512xf32>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 256 : i32, 512 : i32, 1 : i32]
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1024 : i32, 256 : i32, 1 : i32, 1 : i32]
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK: "ttnn.conv2d"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1024 : i32, 512 : i32]
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2>, feature_group_count = 1 : i64, input_dilation = array<i64: 1>, padding = array<i64: 0, 0>, weight_dilation = array<i64: 1>, window_reversal = array<i1: false>, window_strides = array<i64: 1>}> : (tensor<1x256x512xf32>, tensor<1024x256x1xf32>, tensor<1x1024x512xf32>) -> tensor<1x1024x512xf32>
    // CHECK: return %{{.*}} : tensor<1x1024x512xf32, #ttnn_layout3>
    return %1 : tensor<1x1024x512xf32>
  }
}
