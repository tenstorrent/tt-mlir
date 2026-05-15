// RoPE cos subgraph from GPT-OSS-20B (prefill shape: 1x17x32).
// 2 ops, linear chain: cos(freqs) -> multiply(cos, scale).

module {
  func.func @rope_cos_prefill(
      %freqs : tensor<1x17x32xf32>,
      %scale : tensor<1x1x1xf32>)
      -> tensor<1x17x32xf32> {
    %0 = "ttir.cos"(%freqs) : (tensor<1x17x32xf32>) -> tensor<1x17x32xf32>
    %1 = "ttir.multiply"(%0, %scale) : (tensor<1x17x32xf32>, tensor<1x1x1xf32>) -> tensor<1x17x32xf32>
    return %1 : tensor<1x17x32xf32>
  }
}
