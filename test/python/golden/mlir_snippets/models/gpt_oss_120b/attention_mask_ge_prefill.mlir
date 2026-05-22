// Attention mask subgraph from GPT-OSS-120B (d2m_subgraph_2): logical_and,
// ge(si32, si32) -> bf16, logical_and. Uses ge rather than gt.

module {
  func.func @attention_mask_ge_prefill(
      %mask      : tensor<1x1x1x1xbf16>,
      %input_a   : tensor<1x1x17x128xbf16>,
      %indices_a : tensor<1x1x17x1xsi32>,
      %indices_b : tensor<1x1x1x128xsi32>)
      -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.logical_and"(%mask, %input_a) : (tensor<1x1x1x1xbf16>, tensor<1x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    %1 = "ttir.ge"(%indices_a, %indices_b) : (tensor<1x1x17x1xsi32>, tensor<1x1x1x128xsi32>) -> tensor<1x1x17x128xbf16>
    %2 = "ttir.logical_and"(%0, %1) : (tensor<1x1x17x128xbf16>, tensor<1x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %2 : tensor<1x1x17x128xbf16>
  }
}
