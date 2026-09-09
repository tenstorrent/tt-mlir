// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // SDPA backward with causal mask (no attention mask operand).
  func.func @sdpa_bw_causal(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK: "ttnn.sdpa_bw"
    %0, %1, %2 = "ttcore.composite"(%grad_output, %attn_output, %query, %key, %value, %intermediates) <{
        composite_name = "sdpa_bw",
        decomposition = @sdpa_bw_causal_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }

  // SDPA backward with an arbitrary attention mask.
  func.func @sdpa_bw_arbitrary(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>,
      %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK: "ttnn.sdpa_bw"
    %0, %1, %2 = "ttcore.composite"(%grad_output, %attn_output, %query, %key, %value, %intermediates, %mask) <{
        composite_name = "sdpa_bw",
        decomposition = @sdpa_bw_arbitrary_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<arbitrary>,
          dropout_probability = 0.000000e+00 : f32}}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>,
           tensor<1x1x64x64xbf16>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }

  func.func private @sdpa_bw_causal_decomposition(
      %grad_output: tensor<1x8x64x64xbf16>,
      %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>,
      %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
          tensor<1x8x64x64xbf16>) {
    return %query, %key, %value : tensor<1x8x64x64xbf16>,
        tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }

  func.func private @sdpa_bw_arbitrary_decomposition(
      %grad_output: tensor<1x8x64x64xbf16>,
      %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>,
      %intermediates: tensor<1x8x64x32xf32>,
      %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
          tensor<1x8x64x64xbf16>) {
    return %query, %key, %value : tensor<1x8x64x64xbf16>,
        tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}
