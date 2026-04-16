// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=hifi4 max-accuracy=true" %s | FileCheck %s
// Test that the pass preserves existing compute_config values and does NOT override them.
// Note: math_approx_mode=false is forced for SDPA decode when max-accuracy is enabled.

// CHECK-LABEL: func @test_sdpa_decode_preserve_math_fidelity
func.func @test_sdpa_decode_preserve_math_fidelity(%query: tensor<1x32x8x128xbf16>, %key: tensor<32x1x2048x128xbf16>, %value: tensor<32x1x2048x128xbf16>) -> tensor<1x32x8x128xbf16> {
  // CHECK: ttnn.scaled_dot_product_attention_decode
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = lofi, math_approx_mode = false, fp32_dest_acc_en = true>
  // CHECK-NOT: hifi4
  %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value) {
    is_causal = true,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    compute_config = #ttnn.device_compute_kernel_config<math_fidelity = lofi>
  } : (tensor<1x32x8x128xbf16>, tensor<32x1x2048x128xbf16>, tensor<32x1x2048x128xbf16>) -> tensor<1x32x8x128xbf16>

  return %result : tensor<1x32x8x128xbf16>
}
