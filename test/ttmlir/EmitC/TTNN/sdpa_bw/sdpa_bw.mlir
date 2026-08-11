// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  // Causal mask, no attention mask operand.
  // CHECK-LABEL: sdpa_bw_causal
  func.func @sdpa_bw_causal(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK: ttml::metal::sdpa_bw(
    // CHECK-SAME: ::ttml::metal::AttentionMaskType::Causal
    // CHECK-SAME: ::std::nullopt
    // CHECK: ::std::get<0>
    // CHECK: ::std::get<1>
    // CHECK: ::std::get<2>
    %0, %1, %2 = "ttir.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }

  // Arbitrary mask operand present.
  // CHECK-LABEL: sdpa_bw_arbitrary
  func.func @sdpa_bw_arbitrary(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>,
      %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK: ttml::metal::sdpa_bw(
    // CHECK-SAME: ::ttml::metal::AttentionMaskType::Arbitrary
    %0, %1, %2 = "ttir.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates, %mask) <{
        mask_type = #ttcore.attention_mask_type<arbitrary>,
        dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>,
           tensor<1x1x64x64xbf16>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}
