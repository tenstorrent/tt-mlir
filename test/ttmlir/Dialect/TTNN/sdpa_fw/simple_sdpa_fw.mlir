// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // SDPA forward with causal mask, no intermediates.
  func.func @sdpa_fw_causal(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                            %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // CHECK: "ttnn.sdpa_fw"
    %0 = "ttir.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
          -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }

  // SDPA forward with causal mask returning the log-sum-exp intermediates.
  func.func @sdpa_fw_intermediates(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                                   %value: tensor<1x8x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    // CHECK: "ttnn.sdpa_fw"
    %0, %1 = "ttir.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = true}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    return %0, %1 : tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>
  }
}
