// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=lofi fp32-dest-acc-en=true packer-l1-acc=false" %s | FileCheck %s
// Tri-state handling on SDPA ops:
//   - math_fidelity is already hifi4 on the op, so the lofi override must not win.
//   - fp32_dest_acc_en is unset on the op, so the override (true) is applied.
//   - packer_l1_acc is already true on the op, so the false override must not win.

// CHECK-LABEL: func @test_sdpa_preserve_existing
func.func @test_sdpa_preserve_existing(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>, %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
  // CHECK: "ttnn.scaled_dot_product_attention"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true, packer_l1_acc = true>
  // CHECK-NOT: math_fidelity = lofi
  %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) <{
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>,
    is_causal = true,
    compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, packer_l1_acc = true>
  }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
  return %result : tensor<1x8x64x64xbf16>
}

// An explicit `false` from the pipeline is a real value, so it is applied to the
// knobs the op leaves unset while math_fidelity stays at the op's hifi4.
// CHECK-LABEL: func @test_sdpa_decode_merge_with_existing
func.func @test_sdpa_decode_merge_with_existing(%query: tensor<1x32x8x128xbf16>, %key: tensor<32x8x256x128xbf16>, %value: tensor<32x8x256x128xbf16>, %cur_pos: tensor<32xi32>) -> tensor<1x32x8x128xbf16> {
  // CHECK: "ttnn.scaled_dot_product_attention_decode"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true, packer_l1_acc = false>
  %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %cur_pos) <{
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
    is_causal = true,
    compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4>
  }> : (tensor<1x32x8x128xbf16>, tensor<32x8x256x128xbf16>, tensor<32x8x256x128xbf16>, tensor<32xi32>) -> tensor<1x32x8x128xbf16>
  return %result : tensor<1x32x8x128xbf16>
}
