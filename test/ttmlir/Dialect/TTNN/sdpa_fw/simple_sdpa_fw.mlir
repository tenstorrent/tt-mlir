// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // SDPA forward with causal mask, no intermediates.
  func.func @sdpa_fw_causal(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                            %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // CHECK: "ttnn.sdpa_fw"
    %0 = "ttcore.composite"(%query, %key, %value) <{
        composite_name = "sdpa_fw",
        decomposition = @sdpa_fw_causal_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32,
          return_intermediates = false}}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
          -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }

  // SDPA forward with causal mask returning the log-sum-exp intermediates.
  func.func @sdpa_fw_intermediates(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                                   %value: tensor<1x8x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    // CHECK: "ttnn.sdpa_fw"
    %0, %1 = "ttcore.composite"(%query, %key, %value) <{
        composite_name = "sdpa_fw",
        decomposition = @sdpa_fw_intermediates_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32,
          return_intermediates = true}}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    return %0, %1 : tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>
  }

  func.func private @sdpa_fw_causal_decomposition(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    return %query : tensor<1x8x64x64xbf16>
  }

  func.func private @sdpa_fw_intermediates_decomposition(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    %intermediates = "ttir.empty"() : () -> tensor<1x8x64x32xf32>
    return %query, %intermediates : tensor<1x8x64x64xbf16>,
        tensor<1x8x64x32xf32>
  }
}
