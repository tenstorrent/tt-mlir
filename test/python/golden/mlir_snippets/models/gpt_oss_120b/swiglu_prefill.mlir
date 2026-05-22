// SwiGLU / gate-up subgraph from GPT-OSS-120B TP/DP Galaxy batch-128
// (d2m_subgraph_1). Prefill shape 16x2176x2880.

module {
  func.func @swiglu_prefill(
      %value      : tensor<16x2176x2880xbf16>,
      %value_bias : tensor<1x1x1xbf16>,
      %gate       : tensor<16x2176x2880xbf16>,
      %alpha      : tensor<1x1x1xbf16>)
      -> tensor<16x2176x2880xbf16> {
    %0 = "ttir.add"(%value, %value_bias) : (tensor<16x2176x2880xbf16>, tensor<1x1x1xbf16>) -> tensor<16x2176x2880xbf16>
    %1 = "ttir.multiply"(%gate, %alpha) : (tensor<16x2176x2880xbf16>, tensor<1x1x1xbf16>) -> tensor<16x2176x2880xbf16>
    %2 = "ttir.sigmoid"(%1) : (tensor<16x2176x2880xbf16>) -> tensor<16x2176x2880xbf16>
    %3 = "ttir.multiply"(%gate, %2) : (tensor<16x2176x2880xbf16>, tensor<16x2176x2880xbf16>) -> tensor<16x2176x2880xbf16>
    %4 = "ttir.multiply"(%0, %3) : (tensor<16x2176x2880xbf16>, tensor<16x2176x2880xbf16>) -> tensor<16x2176x2880xbf16>
    return %4 : tensor<16x2176x2880xbf16>
  }
}
