// RUN: ttmlir-opt --ttir-element-type-normalization="experimental-bfp8-weights=true" %s | FileCheck %s

// Test that weight operands of conv_transpose2d operations are converted to
// bfp8_b type while activations remain in high precision.

module {
  // CHECK-LABEL: @test_conv_transpose2d_bf16
  func.func @test_conv_transpose2d_bf16(%arg0: tensor<1x8x8x3xbf16>, %arg1: tensor<3x16x3x3xbf16>) -> tensor<1x10x10x16xbf16> {
    // Weight should be cast to bfp8
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<3x16x3x3xbf16>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.conv_transpose2d"(%arg0, %[[weight_cast]], {{.*}}) {{.*}} : (tensor<1x8x8x3xbf16>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x10x10x16xbf16>) -> tensor<1x10x10x16xbf16>
    %0 = ttir.empty() : tensor<1x10x10x16xbf16>
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %0) <{
      stride = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      output_padding = array<i32: 0, 0>,
      dilation = array<i32: 1, 1>,
      groups = 1: i32
    }> : (tensor<1x8x8x3xbf16>, tensor<3x16x3x3xbf16>, tensor<1x10x10x16xbf16>) -> tensor<1x10x10x16xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<1x10x10x16xbf16>
    return %1 : tensor<1x10x10x16xbf16>
  }

  // CHECK-LABEL: @test_conv_transpose2d_f32
  func.func @test_conv_transpose2d_f32(%arg0: tensor<1x8x8x3xf32>, %arg1: tensor<3x16x3x3xf32>) -> tensor<1x10x10x16xf32> {
    // Weight should be cast to bfp8
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<3x16x3x3xf32>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.conv_transpose2d"(%arg0, %[[weight_cast]], {{.*}}) {{.*}} : (tensor<1x8x8x3xf32>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x10x10x16xf32>) -> tensor<1x10x10x16xf32>
    %0 = ttir.empty() : tensor<1x10x10x16xf32>
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %0) <{
      stride = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      output_padding = array<i32: 0, 0>,
      dilation = array<i32: 1, 1>,
      groups = 1: i32
    }> : (tensor<1x8x8x3xf32>, tensor<3x16x3x3xf32>, tensor<1x10x10x16xf32>) -> tensor<1x10x10x16xf32>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<1x10x10x16xf32>
    return %1 : tensor<1x10x10x16xf32>
  }

  // CHECK-LABEL: @test_conv_transpose2d_weight_already_bfp8_arg
  // Edge case: weight is already bfp8 as function argument (should not insert typecast)
  func.func @test_conv_transpose2d_weight_already_bfp8_arg(%arg0: tensor<1x8x8x3xbf16>, %arg1: tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<1x10x10x16xbf16> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.conv_transpose2d"(%arg0, %arg1, {{.*}}) {{.*}} : (tensor<1x8x8x3xbf16>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x10x10x16xbf16>) -> tensor<1x10x10x16xbf16>
    %0 = ttir.empty() : tensor<1x10x10x16xbf16>
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %0) <{
      stride = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      output_padding = array<i32: 0, 0>,
      dilation = array<i32: 1, 1>,
      groups = 1: i32
    }> : (tensor<1x8x8x3xbf16>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x10x10x16xbf16>) -> tensor<1x10x10x16xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<1x10x10x16xbf16>
    return %1 : tensor<1x10x10x16xbf16>
  }

  // CHECK-LABEL: @test_conv_transpose2d_weight_from_typecast_bfp8
  // Edge case: weight is produced by a typecast to bfp8 (should not insert additional typecast)
  func.func @test_conv_transpose2d_weight_from_typecast_bfp8(%arg0: tensor<1x8x8x3xbf16>, %arg1: tensor<3x16x3x3xbf16>) -> tensor<1x10x10x16xbf16> {
    %cast_out = ttir.empty() : tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>
    %weight_cast = "ttir.typecast"(%arg1, %cast_out) : (tensor<3x16x3x3xbf16>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>

    // Should see the one typecast above but not a second one
    // CHECK: %[[typecast:.*]] = "ttir.typecast"
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.conv_transpose2d"(%arg0, %[[typecast]], {{.*}}) {{.*}} : (tensor<1x8x8x3xbf16>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x10x10x16xbf16>) -> tensor<1x10x10x16xbf16>
    %0 = ttir.empty() : tensor<1x10x10x16xbf16>
    %1 = "ttir.conv_transpose2d"(%arg0, %weight_cast, %0) <{
      stride = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      output_padding = array<i32: 0, 0>,
      dilation = array<i32: 1, 1>,
      groups = 1: i32
    }> : (tensor<1x8x8x3xbf16>, tensor<3x16x3x3x!ttcore.tile<32x32, bfp_bf8>>, tensor<1x10x10x16xbf16>) -> tensor<1x10x10x16xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<1x10x10x16xbf16>
    return %1 : tensor<1x10x10x16xbf16>
  }
}
