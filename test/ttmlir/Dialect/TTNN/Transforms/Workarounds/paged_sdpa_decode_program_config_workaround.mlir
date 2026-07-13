// RUN: ttmlir-opt --split-input-file --ttcore-register-device="mock-system-desc-arch=blackhole" --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --check-prefixes=CHECK,BLACKHOLE --input-file=%t
// RUN: ttmlir-opt --split-input-file --ttcore-register-device="mock-system-desc-arch=wormhole_b0" --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --check-prefixes=CHECK,WORMHOLE --input-file=%t

// Blackhole causal, no config: metal defaults + exp_approx_mode = false.
// Wormhole causal, no config: nothing injected.
func.func @sdpa_causal_no_config(
    %arg0: tensor<1x1x4x512xbf16>,
    %arg1: tensor<128x4x32x512xbf16>,
    %arg2: tensor<128x4x32x512xbf16>,
    %arg3: tensor<1x64xi32>,
    %arg4: tensor<1xi32>) -> tensor<1x1x4x512xbf16> {
  // CHECK-LABEL: func.func @sdpa_causal_no_config
  // BLACKHOLE: ttnn.paged_scaled_dot_product_attention_decode
  // BLACKHOLE-SAME: program_config = #ttnn.sdpa_program_config<
  // BLACKHOLE-SAME: q_chunk_size = 32
  // BLACKHOLE-SAME: k_chunk_size = 32
  // BLACKHOLE-SAME: exp_approx_mode = false
  // BLACKHOLE-SAME: max_cores_per_head_batch = 1
  //
  // WORMHOLE: ttnn.paged_scaled_dot_product_attention_decode
  // WORMHOLE-NOT: sdpa_program_config
  %result = "ttnn.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %arg4)
      <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>}>
      : (tensor<1x1x4x512xbf16>, tensor<128x4x32x512xbf16>, tensor<128x4x32x512xbf16>,
         tensor<1x64xi32>, tensor<1xi32>) -> tensor<1x1x4x512xbf16>
  return %result : tensor<1x1x4x512xbf16>
}

// -----

// Non-causal, no config: Blackhole still gets the config; Wormhole gets none
// (metal defaults k_chunk_size = 32 for non-causal decode itself).
func.func @sdpa_non_causal_no_config(
    %arg0: tensor<1x1x4x512xbf16>,
    %arg1: tensor<128x4x32x512xbf16>,
    %arg2: tensor<128x4x32x512xbf16>,
    %arg3: tensor<1x64xi32>,
    %arg4: tensor<1x1x4x512xbf16>,
    %arg5: tensor<1xi32>) -> tensor<1x1x4x512xbf16> {
  // CHECK-LABEL: func.func @sdpa_non_causal_no_config
  // BLACKHOLE: ttnn.paged_scaled_dot_product_attention_decode
  // BLACKHOLE-SAME: program_config = #ttnn.sdpa_program_config<
  // BLACKHOLE-SAME: q_chunk_size = 32
  // BLACKHOLE-SAME: k_chunk_size = 32
  // BLACKHOLE-SAME: exp_approx_mode = false
  // BLACKHOLE-SAME: max_cores_per_head_batch = 1
  //
  // WORMHOLE: ttnn.paged_scaled_dot_product_attention_decode
  // WORMHOLE-NOT: sdpa_program_config
  %result = "ttnn.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
      <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 0>}>
      : (tensor<1x1x4x512xbf16>, tensor<128x4x32x512xbf16>, tensor<128x4x32x512xbf16>,
         tensor<1x64xi32>, tensor<1x1x4x512xbf16>, tensor<1xi32>) -> tensor<1x1x4x512xbf16>
  return %result : tensor<1x1x4x512xbf16>
}

// -----

// Existing config preserved field-by-field; exp_approx_mode forced to false on
// Blackhole, left untouched (true) on Wormhole.
func.func @sdpa_existing_config(
    %arg0: tensor<1x1x4x512xbf16>,
    %arg1: tensor<128x4x32x512xbf16>,
    %arg2: tensor<128x4x32x512xbf16>,
    %arg3: tensor<1x64xi32>,
    %arg4: tensor<1xi32>) -> tensor<1x1x4x512xbf16> {
  // CHECK-LABEL: func.func @sdpa_existing_config
  // CHECK: ttnn.paged_scaled_dot_product_attention_decode
  // CHECK-SAME: program_config = #ttnn.sdpa_program_config<
  // CHECK-SAME: q_chunk_size = 64
  // CHECK-SAME: k_chunk_size = 128
  // BLACKHOLE-SAME: exp_approx_mode = false
  // WORMHOLE-SAME: exp_approx_mode = true
  // CHECK-SAME: max_cores_per_head_batch = 8
  %result = "ttnn.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %arg4)
      <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>,
        program_config = #ttnn.sdpa_program_config<
          compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
          q_chunk_size = 64,
          k_chunk_size = 128,
          exp_approx_mode = true,
          max_cores_per_head_batch = 8>}>
      : (tensor<1x1x4x512xbf16>, tensor<128x4x32x512xbf16>, tensor<128x4x32x512xbf16>,
         tensor<1x64xi32>, tensor<1xi32>) -> tensor<1x1x4x512xbf16>
  return %result : tensor<1x1x4x512xbf16>
}
