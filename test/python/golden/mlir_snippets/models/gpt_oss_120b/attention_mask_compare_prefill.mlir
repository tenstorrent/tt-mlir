// Attention mask compare subgraph from GPT-OSS-120B (d2m_subgraph_72).
// gt(si32, si32) -> bf16, then two logical_ands. Failed trace hoist in #8402
// due to inline ttnn.empty from D2M lowering.

module {
  func.func @attention_mask_compare_prefill(
      %indices_a : tensor<1x1x1x128xsi32>,
      %indices_b : tensor<1x1x17x1xsi32>,
      %mask      : tensor<1x1x1x1xbf16>,
      %input_b   : tensor<1x1x17x128xbf16>)
      -> tensor<1x1x17x128xbf16> {
    %0 = "ttir.gt"(%indices_a, %indices_b) : (tensor<1x1x1x128xsi32>, tensor<1x1x17x1xsi32>) -> tensor<1x1x17x128xbf16>
    %1 = "ttir.logical_and"(%mask, %0) : (tensor<1x1x1x1xbf16>, tensor<1x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    %2 = "ttir.logical_and"(%1, %input_b) : (tensor<1x1x17x128xbf16>, tensor<1x1x17x128xbf16>) -> tensor<1x1x17x128xbf16>
    return %2 : tensor<1x1x17x128xbf16>
  }
}
