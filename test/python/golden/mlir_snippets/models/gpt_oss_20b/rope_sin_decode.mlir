// RoPE sin subgraph from GPT-OSS-20B (decode shape: 1x32x1).
// 2 ops, linear chain: sin(freqs) -> multiply(sin, scale).

module {
  func.func @rope_sin_decode(
      %freqs : tensor<1x32x1xf32>,
      %scale : tensor<1x1x1xf32>)
      -> tensor<1x32x1xf32> {
    %0 = "ttir.sin"(%freqs) : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1 = "ttir.multiply"(%0, %scale) : (tensor<1x32x1xf32>, tensor<1x1x1xf32>) -> tensor<1x32x1xf32>
    return %1 : tensor<1x32x1xf32>
  }
}
