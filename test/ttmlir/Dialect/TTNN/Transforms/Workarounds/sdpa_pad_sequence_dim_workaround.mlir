// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --ttnn-workaround --canonicalize %s | FileCheck %s

module @test_sdpa_workaround attributes {} {
  // Test 1: Workaround should NOT apply - unaligned sequence length with mask
  // (tt-metal now handles sequence padding internally)
  func.func public @test_sdpa_no_workaround_unaligned_seq_with_mask(
    %query: tensor<2x8x31x64xbf16>,
    %key: tensor<2x8x31x64xbf16>,
    %value: tensor<2x8x31x64xbf16>,
    %mask: tensor<2x1x31x31xbf16>
  ) -> tensor<2x8x31x64xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_no_workaround_unaligned_seq_with_mask

    // CHECK-NOT: ttnn.pad
    // CHECK: "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3)
    // CHECK-NOT: ttnn.slice

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false
    }> : (tensor<2x8x31x64xbf16>, tensor<2x8x31x64xbf16>,
         tensor<2x8x31x64xbf16>, tensor<2x1x31x31xbf16>)
      -> tensor<2x8x31x64xbf16>
    return %result : tensor<2x8x31x64xbf16>
  }

  // Test 2: Workaround should NOT apply - no mask
  func.func public @test_sdpa_no_workaround_no_mask(
    %query: tensor<2x8x31x64xbf16>,
    %key: tensor<2x8x31x64xbf16>,
    %value: tensor<2x8x31x64xbf16>
  ) -> tensor<2x8x31x64xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_no_workaround_no_mask

    // CHECK-NOT: ttnn.pad
    // CHECK: "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2)
    // CHECK-NOT: ttnn.slice

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>,
      is_causal = true
    }> : (tensor<2x8x31x64xbf16>, tensor<2x8x31x64xbf16>,
         tensor<2x8x31x64xbf16>)
      -> tensor<2x8x31x64xbf16>
    return %result : tensor<2x8x31x64xbf16>
  }

  // Test 3: Workaround should NOT apply - already aligned
  func.func public @test_sdpa_no_workaround_aligned(
    %query: tensor<2x8x64x64xbf16>,
    %key: tensor<2x8x64x64xbf16>,
    %value: tensor<2x8x64x64xbf16>,
    %mask: tensor<2x1x64x64xbf16>
  ) -> tensor<2x8x64x64xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_no_workaround_aligned

    // CHECK-NOT: ttnn.pad
    // CHECK: "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3)
    // CHECK-NOT: ttnn.slice

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false
    }> : (tensor<2x8x64x64xbf16>, tensor<2x8x64x64xbf16>,
         tensor<2x8x64x64xbf16>, tensor<2x1x64x64xbf16>)
      -> tensor<2x8x64x64xbf16>
    return %result : tensor<2x8x64x64xbf16>
  }

  // Test 4: Workaround SHOULD apply - unaligned head_dim
  func.func public @test_sdpa_workaround_unaligned_head_dim(
    %query: tensor<2x8x64x48xbf16>,
    %key: tensor<2x8x64x48xbf16>,
    %value: tensor<2x8x64x48xbf16>,
    %mask: tensor<2x1x64x64xbf16>
  ) -> tensor<2x8x64x48xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_workaround_unaligned_head_dim

    // CHECK: %[[PADDED_QUERY:[0-9]+]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // CHECK: %[[PADDED_KEY:[0-9]+]] = "ttnn.pad"(%arg1)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // CHECK: %[[PADDED_VALUE:[0-9]+]] = "ttnn.pad"(%arg2)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[PADDED_QUERY]], %[[PADDED_KEY]], %[[PADDED_VALUE]], %arg3)

    // CHECK: %[[SLICED:[0-9]+]] = "ttnn.slice_static"(%[[SDPA]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 8 : i32, 64 : i32, 48 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]

    // CHECK: return %[[SLICED]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false
    }> : (tensor<2x8x64x48xbf16>, tensor<2x8x64x48xbf16>,
         tensor<2x8x64x48xbf16>, tensor<2x1x64x64xbf16>)
      -> tensor<2x8x64x48xbf16>
    return %result : tensor<2x8x64x48xbf16>
  }

  // Test 5: Workaround SHOULD apply - unaligned head_dim without mask
  func.func public @test_sdpa_workaround_unaligned_head_dim_no_mask(
    %query: tensor<2x8x64x48xbf16>,
    %key: tensor<2x8x64x48xbf16>,
    %value: tensor<2x8x64x48xbf16>
  ) -> tensor<2x8x64x48xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_workaround_unaligned_head_dim_no_mask

    // CHECK: %[[PADDED_QUERY:[0-9]+]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // CHECK: %[[PADDED_KEY:[0-9]+]] = "ttnn.pad"(%arg1)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // CHECK: %[[PADDED_VALUE:[0-9]+]] = "ttnn.pad"(%arg2)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[PADDED_QUERY]], %[[PADDED_KEY]], %[[PADDED_VALUE]])

    // CHECK: %[[SLICED:[0-9]+]] = "ttnn.slice_static"(%[[SDPA]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 8 : i32, 64 : i32, 48 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]

    // CHECK: return %[[SLICED]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>,
      is_causal = true
    }> : (tensor<2x8x64x48xbf16>, tensor<2x8x64x48xbf16>,
         tensor<2x8x64x48xbf16>)
      -> tensor<2x8x64x48xbf16>
    return %result : tensor<2x8x64x48xbf16>
  }

  // Test 6: Workaround SHOULD apply - both unaligned seq and head_dim, only
  // head_dim should be padded (sequence padding no longer needed)
  func.func public @test_sdpa_workaround_unaligned_seq_and_head_dim(
    %query: tensor<2x8x31x48xbf16>,
    %key: tensor<2x8x31x48xbf16>,
    %value: tensor<2x8x31x48xbf16>,
    %mask: tensor<2x1x31x31xbf16>
  ) -> tensor<2x8x31x48xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_workaround_unaligned_seq_and_head_dim

    // Only head_dim padding, no sequence padding
    // CHECK: %[[PADDED_QUERY:[0-9]+]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // CHECK: %[[PADDED_KEY:[0-9]+]] = "ttnn.pad"(%arg1)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // CHECK: %[[PADDED_VALUE:[0-9]+]] = "ttnn.pad"(%arg2)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 16>

    // Mask is passed through unchanged
    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[PADDED_QUERY]], %[[PADDED_KEY]], %[[PADDED_VALUE]], %arg3)

    // CHECK: %[[SLICED:[0-9]+]] = "ttnn.slice_static"(%[[SDPA]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 8 : i32, 31 : i32, 48 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]

    // CHECK: return %[[SLICED]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>,
      is_causal = false
    }> : (tensor<2x8x31x48xbf16>, tensor<2x8x31x48xbf16>,
         tensor<2x8x31x48xbf16>, tensor<2x1x31x31xbf16>)
      -> tensor<2x8x31x48xbf16>
    return %result : tensor<2x8x31x48xbf16>
  }

  // Test 7: Decode workaround SHOULD apply - mask num_heads=1 needs broadcast
  // to match query num_heads. tt-metal requires mask[2] == num_heads for decode.
  func.func public @test_sdpa_decode_workaround_broadcast_mask_heads(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>,
    %mask: tensor<1x32x1x128xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_decode_workaround_broadcast_mask_heads

    // Mask should be broadcast from [1, 32, 1, 128] to [1, 32, 32, 128]
    // CHECK: %[[BROADCAST_MASK:[0-9]+]] = "ttnn.repeat"(%arg3)
    // CHECK-SAME: repeat_dims = #ttnn.shape<1x1x32x1>

    // CHECK: "ttnn.scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %[[BROADCAST_MASK]])

    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>, tensor<1x32x1x128xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // Test 8: Decode workaround should NOT apply - mask already has correct num_heads
  func.func public @test_sdpa_decode_no_workaround_mask_heads_match(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>,
    %mask: tensor<1x32x32x128xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_decode_no_workaround_mask_heads_match

    // No repeat needed - mask already has num_heads=32
    // CHECK-NOT: ttnn.repeat
    // CHECK: "ttnn.scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3)

    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %mask) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>, tensor<1x32x32x128xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // Test 9: Decode workaround should NOT apply - no mask
  func.func public @test_sdpa_decode_no_workaround_no_mask(
    %query: tensor<1x32x32x64xbf16>,
    %key: tensor<32x32x128x64xbf16>,
    %value: tensor<32x32x128x64xbf16>
  ) -> tensor<1x32x32x64xbf16> {
    // CHECK-LABEL: func.func public @test_sdpa_decode_no_workaround_no_mask

    // CHECK-NOT: ttnn.repeat
    // CHECK: "ttnn.scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2)

    %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
      is_causal = false,
      scale = 0.125 : f32
    }> : (tensor<1x32x32x64xbf16>, tensor<32x32x128x64xbf16>,
         tensor<32x32x128x64xbf16>)
      -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }
}
