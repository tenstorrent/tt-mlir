// RUN: ttmlir-opt %s | FileCheck %s

module {
  // Test basic layer_norm_post_all_gather with input and stats only
  func.func @forward(%arg0: tensor<1x1x32x128xbf16>, %arg1: tensor<1x1x32x64xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: "ttnn.layer_norm_post_all_gather"
    %0 = "ttnn.layer_norm_post_all_gather"(%arg0, %arg1) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x1x32x128xbf16>
    return %0 : tensor<1x1x32x128xbf16>
  }

  // Test layer_norm_post_all_gather with weight and bias
  func.func @forward_with_weight_bias(%arg0: tensor<1x1x32x128xbf16>, %arg1: tensor<1x1x32x64xbf16>, %arg2: tensor<128xbf16>, %arg3: tensor<128xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: "ttnn.layer_norm_post_all_gather"
    %0 = "ttnn.layer_norm_post_all_gather"(%arg0, %arg1, %arg2, %arg3) <{epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>, tensor<128xbf16>, tensor<128xbf16>) -> tensor<1x1x32x128xbf16>
    return %0 : tensor<1x1x32x128xbf16>
  }

  // Test layer_norm_post_all_gather with dtype attribute
  func.func @forward_with_dtype(%arg0: tensor<1x1x32x128xbf16>, %arg1: tensor<1x1x32x64xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: "ttnn.layer_norm_post_all_gather"
    %0 = "ttnn.layer_norm_post_all_gather"(%arg0, %arg1) <{epsilon = 1.000000e-12 : f32, dtype = #tt.supportedDataTypes<bf16>, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x1x32x128xbf16>
    return %0 : tensor<1x1x32x128xbf16>
  }
}
