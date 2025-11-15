// RUN: ttmlir-opt --ttir-element-type-normalization="experimental-bfp8-weights=true" %s | FileCheck %s

// Test that weight operands (rhs) of dot_general operations are converted to
// bfp8_b type while activations (lhs) remain in high precision.

module {
  // CHECK-LABEL: @test_dot_general_bf16
  func.func @test_dot_general_bf16(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<32x128xbf16> {
    // RHS (weight) should be cast to bfp8
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<64x128xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.dot_general"(%arg0, %[[weight_cast]]) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<32x128xbf16>
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x64xbf16>, tensor<64x128xbf16>) -> tensor<32x128xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    return %0 : tensor<32x128xbf16>
  }

  // CHECK-LABEL: @test_dot_general_f32
  func.func @test_dot_general_f32(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
    // RHS (weight) should be cast to bfp8
    // CHECK: %[[weight_cast:.*]] = "ttir.typecast"(%arg1, {{.*}}) {{.*}} : (tensor<64x128xf32>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    // CHECK: %[[result:.*]] = "ttir.dot_general"(%arg0, %[[weight_cast]]) {{.*}} : (tensor<32x64xf32>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<32x128xf32>
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xf32>
    return %0 : tensor<32x128xf32>
  }

  // CHECK-LABEL: @test_dot_general_weight_already_bfp8_arg
  // Edge case: weight is already bfp8 as function argument (should not insert typecast)
  func.func @test_dot_general_weight_already_bfp8_arg(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<32x128xbf16> {
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.dot_general"(%arg0, %arg1) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<32x128xbf16>
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<32x128xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    return %0 : tensor<32x128xbf16>
  }

  // CHECK-LABEL: @test_dot_general_weight_from_typecast_bfp8
  // Edge case: weight is produced by a typecast to bfp8 (should not insert additional typecast)
  func.func @test_dot_general_weight_from_typecast_bfp8(%arg0: tensor<32x64xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<32x128xbf16> {
    %cast_out = ttir.empty() : tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>
    %weight_cast = "ttir.typecast"(%arg1, %cast_out) : (tensor<64x128xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>

    // Should see the one typecast above but not a second one
    // CHECK: %[[typecast:.*]] = "ttir.typecast"
    // CHECK-NOT: "ttir.typecast"
    // CHECK: %[[result:.*]] = "ttir.dot_general"(%arg0, %[[typecast]]) {{.*}} : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<32x128xbf16>
    %0 = "ttir.dot_general"(%arg0, %weight_cast) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x64xbf16>, tensor<64x128x!ttcore.tile<32x32, bfp_bf8>>) -> tensor<32x128xbf16>
    // Verify output dtype remains unchanged (not converted to bfp8)
    // CHECK: return %[[result]] : tensor<32x128xbf16>
    return %0 : tensor<32x128xbf16>
  }
}
