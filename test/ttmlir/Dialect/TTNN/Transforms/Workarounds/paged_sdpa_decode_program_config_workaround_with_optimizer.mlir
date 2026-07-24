// RUN: ttmlir-opt --split-input-file --ttcore-register-device="mock-system-desc-arch=blackhole" --ttnn-layout --ttnn-workaround="ttnn-optimization-level=1" -o %t %s
// RUN: FileCheck %s --input-file=%t

// exp_approx_mode = false must hold at optimization-level >= 1: the pattern is
// not gated to level 0.

// Blackhole causal, no config: metal defaults + exp_approx_mode = false.
func.func @sdpa_causal_no_config_opt(
    %arg0: tensor<1x1x4x512xbf16>,
    %arg1: tensor<128x4x32x512xbf16>,
    %arg2: tensor<128x4x32x512xbf16>,
    %arg3: tensor<1x64xi32>,
    %arg4: tensor<1xi32>) -> tensor<1x1x4x512xbf16> {
  // CHECK-LABEL: func.func @sdpa_causal_no_config_opt
  // CHECK: ttnn.paged_scaled_dot_product_attention_decode
  // CHECK-SAME: program_config = #ttnn.sdpa_program_config<
  // CHECK-SAME: q_chunk_size = 32
  // CHECK-SAME: k_chunk_size = 32
  // CHECK-SAME: exp_approx_mode = false
  // CHECK-SAME: max_cores_per_head_batch = 1
  %result = "ttnn.paged_scaled_dot_product_attention_decode"(%arg0, %arg1, %arg2, %arg3, %arg4)
      <{is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>}>
      : (tensor<1x1x4x512xbf16>, tensor<128x4x32x512xbf16>, tensor<128x4x32x512xbf16>,
         tensor<1x64xi32>, tensor<1xi32>) -> tensor<1x1x4x512xbf16>
  return %result : tensor<1x1x4x512xbf16>
}

// -----

// Blackhole with a pre-existing config: exp_approx_mode forced to false, chunk
// sizes preserved.
func.func @sdpa_existing_config_opt(
    %arg0: tensor<1x1x4x512xbf16>,
    %arg1: tensor<128x4x32x512xbf16>,
    %arg2: tensor<128x4x32x512xbf16>,
    %arg3: tensor<1x64xi32>,
    %arg4: tensor<1xi32>) -> tensor<1x1x4x512xbf16> {
  // CHECK-LABEL: func.func @sdpa_existing_config_opt
  // CHECK: ttnn.paged_scaled_dot_product_attention_decode
  // CHECK-SAME: program_config = #ttnn.sdpa_program_config<
  // CHECK-SAME: k_chunk_size = 128
  // CHECK-SAME: exp_approx_mode = false
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
