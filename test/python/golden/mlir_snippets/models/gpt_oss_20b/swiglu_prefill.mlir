// SwiGLU / gate-up subgraph from GPT-OSS-20B (prefill shape: 4x544x2880).
// 5 ops with fan-in: (value+bias) and (gate*sigmoid(gate*alpha)) merged by final multiply.

module {
  func.func @swiglu_prefill(
      %value      : tensor<4x544x2880xbf16>,
      %value_bias : tensor<1x1x1xbf16>,
      %gate       : tensor<4x544x2880xbf16>,
      %alpha      : tensor<1x1x1xbf16>)
      -> tensor<4x544x2880xbf16> {
    %0 = "ttir.add"(%value, %value_bias) : (tensor<4x544x2880xbf16>, tensor<1x1x1xbf16>) -> tensor<4x544x2880xbf16>
    %1 = "ttir.multiply"(%gate, %alpha) : (tensor<4x544x2880xbf16>, tensor<1x1x1xbf16>) -> tensor<4x544x2880xbf16>
    %2 = "ttir.sigmoid"(%1) : (tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    %3 = "ttir.multiply"(%gate, %2) : (tensor<4x544x2880xbf16>, tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    %4 = "ttir.multiply"(%0, %3) : (tensor<4x544x2880xbf16>, tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    return %4 : tensor<4x544x2880xbf16>
  }
}
