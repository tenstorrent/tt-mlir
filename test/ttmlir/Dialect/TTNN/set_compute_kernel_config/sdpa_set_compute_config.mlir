// RUN: ttmlir-opt --ttnn-set-compute-kernel-config="math-fidelity=hifi2 fp32-dest-acc-en=true packer-l1-acc=true" %s | FileCheck %s
// Test that the pass applies the global compute-kernel-config override to the
// SDPA family of ops, all of which forward compute_config to ttnn.

// CHECK-LABEL: func @test_sdpa
func.func @test_sdpa(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>, %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
  // CHECK: "ttnn.scaled_dot_product_attention"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>
  %result = "ttnn.scaled_dot_product_attention"(%query, %key, %value) <{
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>,
    is_causal = true
  }> : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
  return %result : tensor<1x8x64x64xbf16>
}

// CHECK-LABEL: func @test_sdpa_decode
func.func @test_sdpa_decode(%query: tensor<1x32x8x128xbf16>, %key: tensor<32x8x256x128xbf16>, %value: tensor<32x8x256x128xbf16>, %cur_pos: tensor<32xi32>) -> tensor<1x32x8x128xbf16> {
  // CHECK: "ttnn.scaled_dot_product_attention_decode"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>
  %result = "ttnn.scaled_dot_product_attention_decode"(%query, %key, %value, %cur_pos) <{
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
    is_causal = true
  }> : (tensor<1x32x8x128xbf16>, tensor<32x8x256x128xbf16>, tensor<32x8x256x128xbf16>, tensor<32xi32>) -> tensor<1x32x8x128xbf16>
  return %result : tensor<1x32x8x128xbf16>
}

// CHECK-LABEL: func @test_paged_sdpa_decode
func.func @test_paged_sdpa_decode(%query: tensor<1x32x8x128xbf16>, %key: tensor<64x8x32x128xbf16>, %value: tensor<64x8x32x128xbf16>, %page_table: tensor<32x2xi32>, %cur_pos: tensor<32xi32>) -> tensor<1x32x8x128xbf16> {
  // CHECK: "ttnn.paged_scaled_dot_product_attention_decode"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>
  %result = "ttnn.paged_scaled_dot_product_attention_decode"(%query, %key, %value, %page_table, %cur_pos) <{
    operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 0>,
    is_causal = true
  }> : (tensor<1x32x8x128xbf16>, tensor<64x8x32x128xbf16>, tensor<64x8x32x128xbf16>, tensor<32x2xi32>, tensor<32xi32>) -> tensor<1x32x8x128xbf16>
  return %result : tensor<1x32x8x128xbf16>
}

// CHECK-LABEL: func @test_chunked_sdpa
func.func @test_chunked_sdpa(%query: tensor<1x8x64x64xbf16>, %key: tensor<64x8x32x64xbf16>, %value: tensor<64x8x32x64xbf16>, %page_table: tensor<1x2xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x8x64x64xbf16> {
  // CHECK: "ttnn.chunked_scaled_dot_product_attention"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>
  %result = "ttnn.chunked_scaled_dot_product_attention"(%query, %key, %value, %page_table, %chunk_start_idx) : (tensor<1x8x64x64xbf16>, tensor<64x8x32x64xbf16>, tensor<64x8x32x64xbf16>, tensor<1x2xi32>, tensor<1xi32>) -> tensor<1x8x64x64xbf16>
  return %result : tensor<1x8x64x64xbf16>
}

// CHECK-LABEL: func @test_flash_mla_prefill
func.func @test_flash_mla_prefill(%query: tensor<1x8x64x192xbf16>, %key: tensor<1x1x64x192xbf16>) -> tensor<1x8x64x128xbf16> {
  // CHECK: "ttnn.flash_mla_prefill"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>
  %result = "ttnn.flash_mla_prefill"(%query, %key) <{
    operandSegmentSizes = array<i32: 1, 1, 0, 0>,
    head_dim_v = 128 : ui32,
    is_causal = true
  }> : (tensor<1x8x64x192xbf16>, tensor<1x1x64x192xbf16>) -> tensor<1x8x64x128xbf16>
  return %result : tensor<1x8x64x128xbf16>
}

// CHECK-LABEL: func @test_paged_flash_mla_decode
func.func @test_paged_flash_mla_decode(%query: tensor<1x32x8x192xbf16>, %key: tensor<64x1x32x192xbf16>, %page_table: tensor<32x2xi32>, %cur_pos: tensor<32xi32>) -> tensor<1x32x8x128xbf16> {
  // CHECK: "ttnn.paged_flash_multi_latent_attention_decode"
  // CHECK-SAME: compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi2, fp32_dest_acc_en = true, packer_l1_acc = true>
  %result = "ttnn.paged_flash_multi_latent_attention_decode"(%query, %key, %page_table, %cur_pos) <{
    operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 1, 0>,
    head_dim_v = 128 : ui32,
    is_causal = true
  }> : (tensor<1x32x8x192xbf16>, tensor<64x1x32x192xbf16>, tensor<32x2xi32>, tensor<32xi32>) -> tensor<1x32x8x128xbf16>
  return %result : tensor<1x32x8x128xbf16>
}
