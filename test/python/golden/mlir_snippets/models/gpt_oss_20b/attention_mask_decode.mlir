// Attention mask subgraph from GPT-OSS-20B (decode shape: 1x1x1x128).
// 2 ops, linear chain: logical_and(mask, input_a) -> logical_and(prev, input_b).

module {
  func.func @attention_mask_decode(
      %mask    : tensor<1x1x1x1xbf16>,
      %input_a : tensor<1x1x1x128xbf16>,
      %input_b : tensor<1x1x1x128xbf16>)
      -> tensor<1x1x1x128xbf16> {
    %0 = "ttir.logical_and"(%mask, %input_a) : (tensor<1x1x1x1xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    %1 = "ttir.logical_and"(%0, %input_b) : (tensor<1x1x1x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x1x1x128xbf16>
    return %1 : tensor<1x1x1x128xbf16>
  }
}
