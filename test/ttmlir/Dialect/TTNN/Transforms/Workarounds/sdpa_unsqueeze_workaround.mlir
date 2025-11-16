// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --ttnn-workaround --canonicalize %s | FileCheck %s

module @test_sdpa_unsqueeze_workaround attributes {} {
  // Test 1: Workaround SHOULD apply - 3D inputs without mask
  func.func public @test_sdpa_unsqueeze_3d_no_mask(
    %query: tensor<8x32x64xf32>,
    %key: tensor<8x32x64xf32>,
    %value: tensor<8x32x64xf32>
  ) -> tensor<8x32x64xf32> {
    // CHECK-LABEL: func.func public @test_sdpa_unsqueeze_3d_no_mask

    // CHECK: %[[QUERY_4D:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[KEY_4D:[0-9]+]] = "ttnn.reshape"(%arg1)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[VALUE_4D:[0-9]+]] = "ttnn.reshape"(%arg2)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[QUERY_4D]], %[[KEY_4D]], %[[VALUE_4D]])

    // CHECK: %[[RESULT_3D:[0-9]+]] = "ttnn.reshape"(%[[SDPA]])
    // CHECK-SAME: shape = [8 : i32, 32 : i32, 64 : i32]

    // CHECK: return %[[RESULT_3D]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) {
      is_causal = true
    } : (tensor<8x32x64xf32>, tensor<8x32x64xf32>,
         tensor<8x32x64xf32>)
      -> tensor<8x32x64xf32>
    return %result : tensor<8x32x64xf32>
  }

  // Test 2: Workaround SHOULD apply - 3D inputs with 3D mask
  func.func public @test_sdpa_unsqueeze_3d_with_3d_mask(
    %query: tensor<8x32x64xf32>,
    %key: tensor<8x32x64xf32>,
    %value: tensor<8x32x64xf32>,
    %mask: tensor<1x1x32x32xf32>
  ) -> tensor<8x32x64xf32> {
    // CHECK-LABEL: func.func public @test_sdpa_unsqueeze_3d_with_3d_mask

    // CHECK: %[[QUERY_4D:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[KEY_4D:[0-9]+]] = "ttnn.reshape"(%arg1)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[VALUE_4D:[0-9]+]] = "ttnn.reshape"(%arg2)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[QUERY_4D]], %[[KEY_4D]], %[[VALUE_4D]]

    // CHECK: %[[RESULT_3D:[0-9]+]] = "ttnn.reshape"(%[[SDPA]])
    // CHECK-SAME: shape = [8 : i32, 32 : i32, 64 : i32]

    // CHECK: return %[[RESULT_3D]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) {
      is_causal = false
    } : (tensor<8x32x64xf32>, tensor<8x32x64xf32>,
         tensor<8x32x64xf32>, tensor<1x1x32x32xf32>)
      -> tensor<8x32x64xf32>
    return %result : tensor<8x32x64xf32>
  }

  // Test 3: Workaround should NOT apply - 4D inputs
  func.func public @test_sdpa_no_unsqueeze_4d_inputs(
    %query: tensor<2x8x32x64xf32>,
    %key: tensor<2x8x32x64xf32>,
    %value: tensor<2x8x32x64xf32>
  ) -> tensor<2x8x32x64xf32> {
    // CHECK-LABEL: func.func public @test_sdpa_no_unsqueeze_4d_inputs

    // CHECK-NOT: ttnn.reshape
    // CHECK: "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2)

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) {
      is_causal = true
    } : (tensor<2x8x32x64xf32>, tensor<2x8x32x64xf32>,
         tensor<2x8x32x64xf32>)
      -> tensor<2x8x32x64xf32>
    return %result : tensor<2x8x32x64xf32>
  }

  // Test 4: Workaround SHOULD apply - 3D inputs with 4D mask (mask should NOT be unsqueezed)
  func.func public @test_sdpa_unsqueeze_3d_with_4d_mask(
    %query: tensor<8x32x64xf32>,
    %key: tensor<8x32x64xf32>,
    %value: tensor<8x32x64xf32>,
    %mask: tensor<1x1x32x32xf32>
  ) -> tensor<8x32x64xf32> {
    // CHECK-LABEL: func.func public @test_sdpa_unsqueeze_3d_with_4d_mask

    // CHECK: %[[QUERY_4D:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[KEY_4D:[0-9]+]] = "ttnn.reshape"(%arg1)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[VALUE_4D:[0-9]+]] = "ttnn.reshape"(%arg2)
    // CHECK-SAME: shape = [1 : i32, 8 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[QUERY_4D]], %[[KEY_4D]], %[[VALUE_4D]], %arg3)

    // CHECK: %[[RESULT_3D:[0-9]+]] = "ttnn.reshape"(%[[SDPA]])
    // CHECK-SAME: shape = [8 : i32, 32 : i32, 64 : i32]

    // CHECK: return %[[RESULT_3D]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) {
      is_causal = false
    } : (tensor<8x32x64xf32>, tensor<8x32x64xf32>,
         tensor<8x32x64xf32>, tensor<1x1x32x32xf32>)
      -> tensor<8x32x64xf32>
    return %result : tensor<8x32x64xf32>
  }

  // Test 5: Workaround SHOULD apply - 3D inputs with scale attribute
  func.func public @test_sdpa_unsqueeze_3d_with_scale(
    %query: tensor<8x32x64xf32>,
    %key: tensor<8x32x64xf32>,
    %value: tensor<8x32x64xf32>
  ) -> tensor<8x32x64xf32> {
    // CHECK-LABEL: func.func public @test_sdpa_unsqueeze_3d_with_scale

    // CHECK: %[[QUERY_4D:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK: %[[KEY_4D:[0-9]+]] = "ttnn.reshape"(%arg1)
    // CHECK: %[[VALUE_4D:[0-9]+]] = "ttnn.reshape"(%arg2)

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[QUERY_4D]], %[[KEY_4D]], %[[VALUE_4D]])
    // CHECK-SAME: scale = {{0.125000|1.250000e-01}}

    // CHECK: %[[RESULT_3D:[0-9]+]] = "ttnn.reshape"(%[[SDPA]])
    // CHECK: return %[[RESULT_3D]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) {
      is_causal = true,
      scale = 0.125 : f32
    } : (tensor<8x32x64xf32>, tensor<8x32x64xf32>,
         tensor<8x32x64xf32>)
      -> tensor<8x32x64xf32>
    return %result : tensor<8x32x64xf32>
  }

  // Test 6: Edge case - different batch sizes in 3D (num_heads varies)
  func.func public @test_sdpa_unsqueeze_3d_varying_heads(
    %query: tensor<16x32x64xf32>,
    %key: tensor<16x32x64xf32>,
    %value: tensor<16x32x64xf32>
  ) -> tensor<16x32x64xf32> {
    // CHECK-LABEL: func.func public @test_sdpa_unsqueeze_3d_varying_heads

    // CHECK: %[[QUERY_4D:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 16 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[KEY_4D:[0-9]+]] = "ttnn.reshape"(%arg1)
    // CHECK-SAME: shape = [1 : i32, 16 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[VALUE_4D:[0-9]+]] = "ttnn.reshape"(%arg2)
    // CHECK-SAME: shape = [1 : i32, 16 : i32, 32 : i32, 64 : i32]

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[QUERY_4D]], %[[KEY_4D]], %[[VALUE_4D]])

    // CHECK: %[[RESULT_3D:[0-9]+]] = "ttnn.reshape"(%[[SDPA]])
    // CHECK-SAME: shape = [16 : i32, 32 : i32, 64 : i32]

    // CHECK: return %[[RESULT_3D]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) {
      is_causal = true
    } : (tensor<16x32x64xf32>, tensor<16x32x64xf32>,
         tensor<16x32x64xf32>)
      -> tensor<16x32x64xf32>
    return %result : tensor<16x32x64xf32>
  }
}
