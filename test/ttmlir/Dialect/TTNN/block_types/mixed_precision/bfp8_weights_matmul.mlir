// RUN: ttmlir-opt --ttir-element-type-normalization="experimental-bfp8-weights=true" %s | FileCheck %s

// Test that weight operands (operand 1) of matmul operations are converted to
// bfp8_b type while activations (operand 0) remain in high precision.

module {
  // CHECK-LABEL: @test_matmul_bf16
  func.func @test_matmul_bf16(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<32x128xbf16> {
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<64x128xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.matmul"(%arg0, %[[weight_cast]], {{.*}}) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    // Verify output dtype remains bf16 (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }

  // CHECK-LABEL: @test_matmul_f32
  func.func @test_matmul_f32(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<64x128xf32>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.matmul"(%arg0, %[[weight_cast]], {{.*}}) {{.*}} : (tensor<32x64xf32>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x128xf32>) -> tensor<32x128xf32>
    // Verify output dtype remains f32 (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xf32>
    %0 = ttir.empty() : tensor<32x128xf32>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<32x64xf32>, tensor<64x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }

  // CHECK-LABEL: @test_matmul_sequential
  // Test that multiple matmuls in sequence all get their weights converted
  func.func @test_matmul_sequential(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128xbf16>, %arg2: tensor<128x256xbf16>) -> tensor<32x256xbf16> {
    // First matmul: weight1 should be cast
    // CHECK: %[[weight1_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<64x128xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result1:.*]] = "ttir.matmul"(%arg0, %[[weight1_cast]], {{.*}}) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>) -> tensor<32x128xbf16>

    // Second matmul: weight2 should be cast
    // CHECK: %[[weight2_cast:.*]] = "ttir.typecast"(%arg2, {{.*}}) {{.*}} : (tensor<128x256xbf16>, tensor<128x256x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<128x256x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result2:.*]] = "ttir.matmul"(%[[result1]], %[[weight2_cast]], {{.*}}) {{.*}} : (tensor<32x128xbf16>, tensor<128x256x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
    %2 = ttir.empty() : tensor<32x256xbf16>
    %3 = "ttir.matmul"(%1, %arg2, %2) : (tensor<32x128xbf16>, tensor<128x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>

    // Verify final output dtype remains bf16 (not converted to bfp8)
    // CHECK: return %[[result2]] : tensor<32x256xbf16>
    return %3 : tensor<32x256xbf16>
  }

  // CHECK-LABEL: @test_matmul_weight_already_bfp8_arg
  // Edge case: weight is already bfp8 as function argument (should not insert typecast)
  func.func @test_matmul_weight_already_bfp8_arg(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<32x128xbf16> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.matmul"(%arg0, %arg1, {{.*}}) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    // Verify output dtype remains bf16 (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }

  // CHECK-LABEL: @test_matmul_weight_from_typecast_bfp8
  // Edge case: weight is produced by a typecast to bfp8 (should not insert additional typecast)
  func.func @test_matmul_weight_from_typecast_bfp8(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<32x128xbf16> {
    %cast_out = ttir.empty() : tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    %weight_cast = "ttir.typecast"(%arg1, %cast_out) : (tensor<64x128xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>

    // Should see the one typecast above but not a second one
    // CHECK: %[[typecast:.*]] = "ttir.typecast"
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.matmul"(%arg0, %[[typecast]], {{.*}}) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    // Verify output dtype remains bf16 (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    %0 = ttir.empty() : tensor<32x128xbf16>
    %1 = "ttir.matmul"(%arg0, %weight_cast, %0) : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>, tensor<32x128xbf16>) -> tensor<32x128xbf16>
    return %1 : tensor<32x128xbf16>
  }
}
