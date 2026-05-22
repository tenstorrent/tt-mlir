// Cross-type compare + multiply from GPT-OSS-120B TP/DP Galaxy batch-128
// (d2m_subgraph_0). si32 indices broadcast to bf16 mask, then multiply.
// See issue #8402.

module {
  func.func @compare_eq_multiply(
      %indices_a : tensor<1x4x544xsi32>,
      %indices_b : tensor<16x1x1xsi32>,
      %input     : tensor<1x4x544xbf16>)
      -> tensor<16x4x544xbf16> {
    %0 = "ttir.eq"(%indices_a, %indices_b) : (tensor<1x4x544xsi32>, tensor<16x1x1xsi32>) -> tensor<16x4x544xbf16>
    %1 = "ttir.multiply"(%0, %input) : (tensor<16x4x544xbf16>, tensor<1x4x544xbf16>) -> tensor<16x4x544xbf16>
    return %1 : tensor<16x4x544xbf16>
  }
}
