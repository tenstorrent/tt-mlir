// RUN: ttmlir-opt --ttir-element-type-normalization="experimental-bfp8-weights=true" %s | FileCheck %s

// Test that weight operands of linear operations are converted to bfp8_b type
// while activations and bias remain in high precision.
// NOTE: We only cast weights, NOT bias tensors.

module {
  // CHECK-LABEL: @test_linear_bf16_with_bias
  func.func @test_linear_bf16_with_bias(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128xbf16>, %arg2: tensor<128xbf16>) -> tensor<32x128xbf16> {
    // Weight should be cast to bfp8, but bias should remain bf16
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<64x128xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.linear"(%arg0, %[[weight_cast]], %arg2, {{.*}}) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %arg2, %0) : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }

  // CHECK-LABEL: @test_linear_f32_with_bias
  func.func @test_linear_f32_with_bias(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<128xf32>) -> tensor<32x128xf32> {
    // Weight should be cast to bfp8, but bias should remain f32
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<64x128xf32>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.linear"(%arg0, %[[weight_cast]], %arg2, {{.*}}) {{.*}} : (tensor<32x64xf32>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    %0 = ttir.empty() : tensor<32x128xf32>
    %1 = "ttir.linear"(%arg0, %arg1, %arg2, %0) : (tensor<32x64xf32>, tensor<64x128xf32>, tensor<128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }

  // CHECK-LABEL: @test_linear_without_bias
  func.func @test_linear_without_bias(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<32x128xbf16> {
    // Weight should be cast to bfp8
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<64x128xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.linear"(%arg0, %[[weight_cast]], {{.*}}) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }

  // CHECK-LABEL: @test_linear_weight_already_bfp8_arg
  // Edge case: weight is already bfp8 as function argument (should not insert typecast)
  func.func @test_linear_weight_already_bfp8_arg(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, %arg2: tensor<128xbf16>) -> tensor<32x128xbf16> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.linear"(%arg0, %arg1, %arg2, {{.*}}) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = "ttir.linear"(%arg0, %arg1, %arg2, %0) : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }

  // CHECK-LABEL: @test_linear_weight_from_typecast_bfp8
  // Edge case: weight is produced by a typecast to bfp8 (should not insert additional typecast)
  func.func @test_linear_weight_from_typecast_bfp8(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128xbf16>, %arg2: tensor<128xbf16>) -> tensor<32x128xbf16> {
    %cast_out = ttir.empty() : tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    %weight_cast = "ttir.typecast"(%arg1, %cast_out) : (tensor<64x128xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>

    // Should see the one typecast above but not a second one
    // CHECK: %[[typecast:.*]] = "ttir.typecast"
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.linear"(%arg0, %[[typecast]], %arg2, {{.*}}) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = "ttir.linear"(%arg0, %weight_cast, %arg2, %0) : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }
}
