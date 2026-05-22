// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_13/28).
// 2D scalar scaling chain: (a * s0) * s1.

module {
  func.func @mul_mul_2d(
      %arg0: tensor<512x8xbf16>,
      %arg1: tensor<1x1xbf16>,
      %arg2: tensor<1x1xbf16>) -> tensor<512x8xbf16> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<512x8xbf16>, tensor<1x1xbf16>) -> tensor<512x8xbf16>
    %1 = "ttir.multiply"(%0, %arg2) : (tensor<512x8xbf16>, tensor<1x1xbf16>) -> tensor<512x8xbf16>
    return %1 : tensor<512x8xbf16>
  }
}
