// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=hifi4" %s | FileCheck %s
// Test that math-fidelity=hifi4 alone (without max-accuracy=true) does NOT force
// math_approx_mode=false or exp_approx_mode=false on SDPA decode ops.
// The per-op accuracy rules only apply when max-accuracy is explicitly enabled.

// CHECK-LABEL: func @test_sdpa_decode_hifi4_no_max_accuracy
func.func @test_sdpa_decode_hifi4_no_max_accuracy(%query: tensor<1x32x8x128xbf16>, %key: tensor<32x1x2048x128xbf16>, %value: tensor<32x1x2048x128xbf16>) -> tensor<1x32x8x128xbf16> {
  // CHECK: ttnn.scaled_dot_product_attention_decode
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>
  // CHECK-NOT: math_approx_mode
  // exp_approx_mode should remain true (not overridden)
  // CHECK-SAME: program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = <8, 8>, q_chunk_size = 128, k_chunk_size = 128, exp_approx_mode = true>
  %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value) {
    is_causal = true,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128, exp_approx_mode = true>
  } : (tensor<1x32x8x128xbf16>, tensor<32x1x2048x128xbf16>, tensor<32x1x2048x128xbf16>) -> tensor<1x32x8x128xbf16>

  return %result : tensor<1x32x8x128xbf16>
}
