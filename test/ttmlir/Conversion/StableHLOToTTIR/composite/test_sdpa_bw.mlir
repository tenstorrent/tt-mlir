// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// SDPA backward composite with causal mask (6 operands, 3 results).
module {
  func.func @sdpa_bw(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK: "ttcore.composite"
    // CHECK-SAME: mask_type = #ttcore.attention_mask_type<causal>
    // CHECK-SAME: composite_name = "sdpa_bw"
    // CHECK-NOT: stablehlo.composite
    %0:3 = stablehlo.composite "tenstorrent.sdpa_bw" %grad_output, %attn_output, %query, %key, %value, %intermediates {
      composite_attributes = {
        mask_type = 1 : i32,
        dropout_probability = 0.000000e+00 : f32
      },
      decomposition = @tenstorrent.sdpa_bw.impl
    } : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
  func.func private @tenstorrent.sdpa_bw.impl(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    return %query, %key, %value : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}

// -----

// SDPA backward composite with an arbitrary mask (7 operands, 3 results).
module {
  func.func @sdpa_bw_arbitrary(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>,
      %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK: "ttcore.composite"
    // CHECK-SAME: mask_type = #ttcore.attention_mask_type<arbitrary>
    // CHECK-SAME: composite_name = "sdpa_bw"
    // CHECK-NOT: stablehlo.composite
    %0:3 = stablehlo.composite "tenstorrent.sdpa_bw" %grad_output, %attn_output, %query, %key, %value, %intermediates, %mask {
      composite_attributes = {
        mask_type = 2 : i32,
        dropout_probability = 0.000000e+00 : f32
      },
      decomposition = @tenstorrent.sdpa_bw.arbitrary.impl
    } : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>, tensor<1x1x64x64xbf16>) -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
  func.func private @tenstorrent.sdpa_bw.arbitrary.impl(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>,
      %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    return %query, %key, %value : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}
