// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=hifi2" %s | FileCheck %s

// CHECK-LABEL: func @test_sdpa_decode_without_compute_config
func.func @test_sdpa_decode_without_compute_config(%query: tensor<1x32x8x128xbf16>, %key: tensor<32x1x2048x128xbf16>, %value: tensor<32x1x2048x128xbf16>) -> tensor<1x32x8x128xbf16> {
  // CHECK: ttnn.scaled_dot_product_attention_decode
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true>
  %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value) {
    is_causal = true,
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>
  } : (tensor<1x32x8x128xbf16>, tensor<32x1x2048x128xbf16>, tensor<32x1x2048x128xbf16>) -> tensor<1x32x8x128xbf16>

  return %result : tensor<1x32x8x128xbf16>
}
