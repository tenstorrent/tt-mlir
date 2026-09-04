// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  // Causal mask, no attention mask operand, single (output) result.
  // CHECK-LABEL: sdpa_fw_causal
  func.func @sdpa_fw_causal(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                            %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // CHECK: ttml::metal::sdpa_fw(
    // CHECK-SAME: ::ttml::metal::AttentionMaskType::Causal
    // CHECK-SAME: ::std::nullopt
    // CHECK: util_get_optional_value(
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

  // Arbitrary mask + log-sum-exp intermediates: both results are unpacked.
  // CHECK-LABEL: sdpa_fw_arbitrary_intermediates
  func.func @sdpa_fw_arbitrary_intermediates(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    // CHECK: ttml::metal::sdpa_fw(
    // CHECK-SAME: ::ttml::metal::AttentionMaskType::Arbitrary
    // CHECK-COUNT-2: util_get_optional_value(
    %0, %1 = "ttcore.composite"(%query, %key, %value, %mask) <{
        composite_name = "sdpa_fw",
        decomposition = @sdpa_fw_arbitrary_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<arbitrary>,
          dropout_probability = 0.000000e+00 : f32,
          return_intermediates = true}}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    return %0, %1 : tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>
  }

  func.func private @sdpa_fw_causal_decomposition(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    return %query : tensor<1x8x64x64xbf16>
  }

  func.func private @sdpa_fw_arbitrary_decomposition(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    %intermediates = "ttir.empty"() : () -> tensor<1x8x64x32xf32>
    return %query, %intermediates : tensor<1x8x64x64xbf16>,
        tensor<1x8x64x32xf32>
  }
}
