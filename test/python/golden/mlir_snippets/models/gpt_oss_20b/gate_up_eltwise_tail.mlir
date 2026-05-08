// =============================================================================
// GPT-OSS-20B gate-up SwiGLU-style elementwise tail (post matmul / clamp /
// slice). Fusing this chain into a single d2m.generic via D2M elementwise
// fusion previously hit a read-after-write hazard and produced pcc ~0.5.
// =============================================================================

module {
  func.func @gpt_oss_20b_gate_up_eltwise_tail(
      %value      : tensor<4x544x2880xbf16>,
      %value_bias : tensor<1x1x1xbf16>,
      %gate       : tensor<4x544x2880xbf16>,
      %alpha      : tensor<1x1x1xbf16>)
      -> tensor<4x544x2880xbf16> {
    %0 = "ttir.add"(%value, %value_bias)
        : (tensor<4x544x2880xbf16>, tensor<1x1x1xbf16>) -> tensor<4x544x2880xbf16>
    %1 = "ttir.multiply"(%gate, %alpha)
        : (tensor<4x544x2880xbf16>, tensor<1x1x1xbf16>) -> tensor<4x544x2880xbf16>
    %2 = "ttir.sigmoid"(%1)
        : (tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    %3 = "ttir.multiply"(%gate, %2)
        : (tensor<4x544x2880xbf16>, tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    %4 = "ttir.multiply"(%0, %3)
        : (tensor<4x544x2880xbf16>, tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    return %4 : tensor<4x544x2880xbf16>
  }
}
