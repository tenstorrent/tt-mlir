// RUN: ttmlir-opt --ttir-element-type-normalization="experimental-bfp8-weights=true" %s | FileCheck %s

// Test that weight operands of convolution operations are converted to
// bfp8_b type while activations remain in high precision.

module {
  // CHECK-LABEL: @test_convolution_bf16
  func.func @test_convolution_bf16(%arg0: tensor<1x3x32x32xbf16>, %arg1: tensor<16x3x3x3xbf16>) -> tensor<1x16x30x30xbf16> {
    // Weight should be cast to bfp8
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<16x3x3x3xbf16>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.convolution"(%arg0, %[[weight_cast]], {{.*}}) {{.*}} : (tensor<1x3x32x32xbf16>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16>
    %0 = ttir.empty() : tensor<1x16x30x30xbf16>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<
        convolution_layout input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3>,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 0, 0, 0, 0>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x32x32xbf16>, tensor<16x3x3x3xbf16>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<1x16x30x30xbf16>
    return %1 : tensor<1x16x30x30xbf16>
  }

  // CHECK-LABEL: @test_convolution_f32
  func.func @test_convolution_f32(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<16x3x3x3xf32>) -> tensor<1x16x30x30xf32> {
    // Weight should be cast to bfp8
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<16x3x3x3xf32>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.convolution"(%arg0, %[[weight_cast]], {{.*}}) {{.*}} : (tensor<1x3x32x32xf32>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
    %0 = ttir.empty() : tensor<1x16x30x30xf32>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<
        convolution_layout input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3>,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 0, 0, 0, 0>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>, tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<1x16x30x30xf32>
    return %1 : tensor<1x16x30x30xf32>
  }

  // CHECK-LABEL: @test_convolution_weight_already_bfp8_arg
  // Edge case: weight is already bfp8 as function argument (should not insert typecast)
  func.func @test_convolution_weight_already_bfp8_arg(%arg0: tensor<1x3x32x32xbf16>, %arg1: tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<1x16x30x30xbf16> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.convolution"(%arg0, %arg1, {{.*}}) {{.*}} : (tensor<1x3x32x32xbf16>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16>
    %0 = ttir.empty() : tensor<1x16x30x30xbf16>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<
        convolution_layout input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3>,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 0, 0, 0, 0>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x32x32xbf16>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<1x16x30x30xbf16>
    return %1 : tensor<1x16x30x30xbf16>
  }

  // CHECK-LABEL: @test_convolution_weight_from_typecast_bfp8
  // Edge case: weight is produced by a typecast to bfp8 (should not insert additional typecast)
  func.func @test_convolution_weight_from_typecast_bfp8(%arg0: tensor<1x3x32x32xbf16>, %arg1: tensor<16x3x3x3xbf16>) -> tensor<1x16x30x30xbf16> {
    %cast_out = ttir.empty() : tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>
    %weight_cast = "ttir.typecast"(%arg1, %cast_out) : (tensor<16x3x3x3xbf16>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>

    // Should see the one typecast above but not a second one
    // CHECK: %[[typecast:.*]] = "ttir.typecast"
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.convolution"(%arg0, %[[typecast]], {{.*}}) {{.*}} : (tensor<1x3x32x32xbf16>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16>
    %0 = ttir.empty() : tensor<1x16x30x30xbf16>
    %1 = "ttir.convolution"(%arg0, %weight_cast, %0) <{
      batch_group_count = 1 : i64,
      convolution_layout = #ttir<
        convolution_layout input_batch = 0,
        input_feature = 1,
        input_spatial_dimensions = 2x3,
        kernel_output_feature = 0,
        kernel_input_feature = 1,
        kernel_spatial_dimensions = 2x3,
        output_batch = 0,
        output_feature = 1,
        output_spatial_dimensions = 2x3>,
      feature_group_count = 1 : i64,
      input_dilation = array<i64: 1, 1>,
      padding = array<i64: 0, 0, 0, 0>,
      weight_dilation = array<i64: 1, 1>,
      window_reversal = array<i1: false, false>,
      window_strides = array<i64: 1, 1>
    }> : (tensor<1x3x32x32xbf16>, tensor<16x3x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x16x30x30xbf16>) -> tensor<1x16x30x30xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<1x16x30x30xbf16>
    return %1 : tensor<1x16x30x30xbf16>
  }
}
