// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module @test_sdpa_workaround attributes {} {
  // Test 1: Workaround SHOULD apply - mask present, unaligned sequence length
  func.func public @test_sdpa_workaround_with_mask_unaligned(
    %query: tensor<2x8x31x64xf32, #ttnn_layout>,
    %key: tensor<2x8x31x64xf32, #ttnn_layout>,
    %value: tensor<2x8x31x64xf32, #ttnn_layout>,
    %mask: tensor<2x1x31x31xf32, #ttnn_layout>
  ) -> tensor<2x8x31x64xf32, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_sdpa_workaround_with_mask_unaligned

    // CHECK: %[[PADDED_QUERY:[0-9]+]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 1, 0, 0>
    // CHECK-SAME: (tensor<2x8x31x64xf32, #ttnn_layout>) -> tensor<2x8x32x64xf32, #ttnn_layout>

    // CHECK: %[[PADDED_MASK:[0-9]+]] = "ttnn.pad"(%arg3)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 1, 0, 0>
    // CHECK-SAME: (tensor<2x1x31x31xf32, #ttnn_layout>) -> tensor<2x1x32x31xf32, #ttnn_layout>

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[PADDED_QUERY]], %arg1, %arg2, %[[PADDED_MASK]])
    // CHECK-SAME: tensor<2x8x32x64xf32, #ttnn_layout>

    // CHECK: %[[SLICED:[0-9]+]] = "ttnn.slice_static"(%[[SDPA]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 8 : i32, 31 : i32, 64 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    // CHECK-SAME: (tensor<2x8x32x64xf32, #ttnn_layout>) -> tensor<2x8x31x64xf32, #ttnn_layout>

    // CHECK: return %[[SLICED]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) {
      is_causal = false
    } : (tensor<2x8x31x64xf32, #ttnn_layout>, tensor<2x8x31x64xf32, #ttnn_layout>,
         tensor<2x8x31x64xf32, #ttnn_layout>, tensor<2x1x31x31xf32, #ttnn_layout>)
      -> tensor<2x8x31x64xf32, #ttnn_layout>
    return %result : tensor<2x8x31x64xf32, #ttnn_layout>
  }

  // Test 2: Workaround should NOT apply - no mask
  func.func public @test_sdpa_no_workaround_no_mask(
    %query: tensor<2x8x31x64xf32, #ttnn_layout>,
    %key: tensor<2x8x31x64xf32, #ttnn_layout>,
    %value: tensor<2x8x31x64xf32, #ttnn_layout>
  ) -> tensor<2x8x31x64xf32, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_sdpa_no_workaround_no_mask

    // CHECK-NOT: ttnn.pad
    // CHECK: "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2)
    // CHECK-NOT: ttnn.slice

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) {
      is_causal = true
    } : (tensor<2x8x31x64xf32, #ttnn_layout>, tensor<2x8x31x64xf32, #ttnn_layout>,
         tensor<2x8x31x64xf32, #ttnn_layout>)
      -> tensor<2x8x31x64xf32, #ttnn_layout>
    return %result : tensor<2x8x31x64xf32, #ttnn_layout>
  }

  // Test 3: Workaround should NOT apply - already aligned
  func.func public @test_sdpa_no_workaround_aligned(
    %query: tensor<2x8x64x64xf32, #ttnn_layout>,
    %key: tensor<2x8x64x64xf32, #ttnn_layout>,
    %value: tensor<2x8x64x64xf32, #ttnn_layout>,
    %mask: tensor<2x1x64x64xf32, #ttnn_layout>
  ) -> tensor<2x8x64x64xf32, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_sdpa_no_workaround_aligned

    // CHECK-NOT: ttnn.pad
    // CHECK: "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3)
    // CHECK-NOT: ttnn.slice

    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) {
      is_causal = false
    } : (tensor<2x8x64x64xf32, #ttnn_layout>, tensor<2x8x64x64xf32, #ttnn_layout>,
         tensor<2x8x64x64xf32, #ttnn_layout>, tensor<2x1x64x64xf32, #ttnn_layout>)
      -> tensor<2x8x64x64xf32, #ttnn_layout>
    return %result : tensor<2x8x64x64xf32, #ttnn_layout>
  }

  // Test 4: Edge case - very small sequence length (seq_len = 1)
  func.func public @test_sdpa_workaround_seq_len_1(
    %query: tensor<2x8x1x64xf32, #ttnn_layout>,
    %key: tensor<2x8x1x64xf32, #ttnn_layout>,
    %value: tensor<2x8x1x64xf32, #ttnn_layout>,
    %mask: tensor<2x1x1x1xf32, #ttnn_layout>
  ) -> tensor<2x8x1x64xf32, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_sdpa_workaround_seq_len_1

    // CHECK: %[[PADDED_QUERY:[0-9]+]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 31, 0, 0>
    // CHECK-SAME: (tensor<2x8x1x64xf32, #ttnn_layout>) -> tensor<2x8x32x64xf32, #ttnn_layout>

    // CHECK: %[[PADDED_MASK:[0-9]+]] = "ttnn.pad"(%arg3)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 31, 0, 0>
    // CHECK-SAME: (tensor<2x1x1x1xf32, #ttnn_layout>) -> tensor<2x1x32x1xf32, #ttnn_layout>

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[PADDED_QUERY]], %arg1, %arg2, %[[PADDED_MASK]])
    // CHECK-SAME: tensor<2x8x32x64xf32, #ttnn_layout>

    // CHECK: %[[SLICED:[0-9]+]] = "ttnn.slice_static"(%[[SDPA]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 8 : i32, 1 : i32, 64 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    // CHECK-SAME: (tensor<2x8x32x64xf32, #ttnn_layout>) -> tensor<2x8x1x64xf32, #ttnn_layout>

    // CHECK: return %[[SLICED]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) {
      is_causal = false
    } : (tensor<2x8x1x64xf32, #ttnn_layout>, tensor<2x8x1x64xf32, #ttnn_layout>,
         tensor<2x8x1x64xf32, #ttnn_layout>, tensor<2x1x1x1xf32, #ttnn_layout>)
      -> tensor<2x8x1x64xf32, #ttnn_layout>
    return %result : tensor<2x8x1x64xf32, #ttnn_layout>
  }

  // Test 5: Edge case - just over boundary (seq_len = 33)
  func.func public @test_sdpa_workaround_seq_len_33(
    %query: tensor<2x8x33x64xf32, #ttnn_layout>,
    %key: tensor<2x8x33x64xf32, #ttnn_layout>,
    %value: tensor<2x8x33x64xf32, #ttnn_layout>,
    %mask: tensor<2x1x33x33xf32, #ttnn_layout>
  ) -> tensor<2x8x33x64xf32, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_sdpa_workaround_seq_len_33

    // CHECK: %[[PADDED_QUERY:[0-9]+]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 31, 0, 0>
    // CHECK-SAME: (tensor<2x8x33x64xf32, #ttnn_layout>) -> tensor<2x8x64x64xf32, #ttnn_layout>

    // CHECK: %[[PADDED_MASK:[0-9]+]] = "ttnn.pad"(%arg3)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 31, 0, 0>
    // CHECK-SAME: (tensor<2x1x33x33xf32, #ttnn_layout>) -> tensor<2x1x64x33xf32, #ttnn_layout>

    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[PADDED_QUERY]], %arg1, %arg2, %[[PADDED_MASK]])
    // CHECK-SAME: tensor<2x8x64x64xf32, #ttnn_layout>

    // CHECK: %[[SLICED:[0-9]+]] = "ttnn.slice_static"(%[[SDPA]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 8 : i32, 33 : i32, 64 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    // CHECK-SAME: (tensor<2x8x64x64xf32, #ttnn_layout>) -> tensor<2x8x33x64xf32, #ttnn_layout>

    // CHECK: return %[[SLICED]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) {
      is_causal = false
    } : (tensor<2x8x33x64xf32, #ttnn_layout>, tensor<2x8x33x64xf32, #ttnn_layout>,
         tensor<2x8x33x64xf32, #ttnn_layout>, tensor<2x1x33x33xf32, #ttnn_layout>)
      -> tensor<2x8x33x64xf32, #ttnn_layout>
    return %result : tensor<2x8x33x64xf32, #ttnn_layout>
  }

  // Test 6: Cross-attention - different sequence lengths for Q vs K/V
  func.func public @test_sdpa_workaround_cross_attention(
    %query: tensor<2x8x31x64xf32, #ttnn_layout>,
    %key: tensor<2x8x64x64xf32, #ttnn_layout>,
    %value: tensor<2x8x64x64xf32, #ttnn_layout>,
    %mask: tensor<2x1x31x64xf32, #ttnn_layout>
  ) -> tensor<2x8x31x64xf32, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_sdpa_workaround_cross_attention

    // Query has unaligned seq_len (31), K/V have aligned seq_len (64)
    // Only query and mask should be padded, K/V remain unchanged

    // CHECK: %[[PADDED_QUERY:[0-9]+]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 1, 0, 0>
    // CHECK-SAME: (tensor<2x8x31x64xf32, #ttnn_layout>) -> tensor<2x8x32x64xf32, #ttnn_layout>

    // CHECK: %[[PADDED_MASK:[0-9]+]] = "ttnn.pad"(%arg3)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 1, 0, 0>
    // CHECK-SAME: (tensor<2x1x31x64xf32, #ttnn_layout>) -> tensor<2x1x32x64xf32, #ttnn_layout>

    // K and V (%arg1, %arg2) should be used directly without padding
    // CHECK: %[[SDPA:[0-9]+]] = "ttnn.scaled_dot_product_attention"(%[[PADDED_QUERY]], %arg1, %arg2, %[[PADDED_MASK]])
    // CHECK-SAME: tensor<2x8x32x64xf32, #ttnn_layout>

    // CHECK: %[[SLICED:[0-9]+]] = "ttnn.slice_static"(%[[SDPA]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 8 : i32, 31 : i32, 64 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]
    // CHECK-SAME: (tensor<2x8x32x64xf32, #ttnn_layout>) -> tensor<2x8x31x64xf32, #ttnn_layout>

    // CHECK: return %[[SLICED]]
    %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value, %mask) {
      is_causal = false
    } : (tensor<2x8x31x64xf32, #ttnn_layout>, tensor<2x8x64x64xf32, #ttnn_layout>,
         tensor<2x8x64x64xf32, #ttnn_layout>, tensor<2x1x31x64xf32, #ttnn_layout>)
      -> tensor<2x8x31x64xf32, #ttnn_layout>
    return %result : tensor<2x8x31x64xf32, #ttnn_layout>
  }
}
