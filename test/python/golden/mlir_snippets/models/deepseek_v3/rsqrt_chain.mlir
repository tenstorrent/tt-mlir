// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_1/6/19/21/34).
// RMSNorm-style epsilon chain: rsqrt((a * b) + c) with scalar broadcasts.

module {
  func.func @rsqrt_chain(
      %arg0: tensor<32x16xf32>,
      %arg1: tensor<1x1xf32>,
      %arg2: tensor<1x1xf32>) -> tensor<32x16xf32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x16xf32>, tensor<1x1xf32>) -> tensor<32x16xf32>
    %1 = "ttir.add"(%0, %arg2) : (tensor<32x16xf32>, tensor<1x1xf32>) -> tensor<32x16xf32>
    %2 = "ttir.rsqrt"(%1) : (tensor<32x16xf32>) -> tensor<32x16xf32>
    return %2 : tensor<32x16xf32>
  }
}
