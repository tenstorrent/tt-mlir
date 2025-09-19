// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @conv1d_test1(%arg0: tensor<1x256x512xf32>, %arg1: tensor<1024x256x1xf32>, %arg2: tensor<1024xf32>) -> tensor<1x1024x512xf32> {
    %0 = ttir.empty() : tensor<1x1024x512xf32>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 256 : i32, 512 : i32, 1 : i32]
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1024 : i32, 256 : i32, 1 : i32, 1 : i32]
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 2, 3, 1>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [512 : i32, 256 : i32]
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 1, 0, 2, 3>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [256 : i32, 1024 : i32]
    // CHECK: "ttnn.matmul"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 512 : i32, 1 : i32, 1024 : i32]
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 1024 : i32, 512 : i32]
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2>, feature_group_count = 1 : i64, input_dilation = array<i64: 1>, padding = array<i64: 0, 0>, weight_dilation = array<i64: 1>, window_reversal = array<i1: false>, window_strides = array<i64: 1>}> : (tensor<1x256x512xf32>, tensor<1024x256x1xf32>, tensor<1x1024x512xf32>) -> tensor<1x1024x512xf32>
    // CHECK: return %{{.*}} : tensor<1x1024x512xf32, #ttnn_layout3>
    return %1 : tensor<1x1024x512xf32>
  }

  // Test a different ordering of dimensions
  func.func public @conv1d_test2(%arg0: tensor<1x7x768xbf16>, %arg1: tensor<1x192x768xbf16>) -> (tensor<1x7x768xbf16>) {
    %0 = ttir.empty() : tensor<1x7x768xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 7 : i32, 768 : i32, 1 : i32]
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 192 : i32, 768 : i32, 1 : i32]
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 1, 3, 2>
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 1, 0, 3>
    // CHECK: "ttnn.conv2d"
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 1, 3, 2>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 7 : i32, 768 : i32]
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 2, input_spatial_dimensions = 1, kernel_output_feature = 2, kernel_input_feature = 1, kernel_spatial_dimensions = 0, output_batch = 0, output_feature = 2, output_spatial_dimensions = 1>, feature_group_count = 4 : i64, input_dilation = array<i64: 1>, padding = array<i64: 0, 0>, weight_dilation = array<i64: 1>, window_reversal = array<i1: false>, window_strides = array<i64: 1>}> : (tensor<1x7x768xbf16>, tensor<1x192x768xbf16>, tensor<1x7x768xbf16>) -> tensor<1x7x768xbf16>
    // CHECK: return %{{.*}} : tensor<1x7x768xbf16, #ttnn_layout{{.*}}>
    return %1 : tensor<1x7x768xbf16>
  }
}
