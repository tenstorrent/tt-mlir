// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_4).
// Integer FMA-style chain: (a * b) + c on si32 with column-broadcast scalar.

module {
  func.func @mul_add_si32(
      %arg0: tensor<8192x1xsi32>,
      %arg1: tensor<1x1xsi32>,
      %arg2: tensor<8192x1xsi32>) -> tensor<8192x1xsi32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<8192x1xsi32>, tensor<1x1xsi32>) -> tensor<8192x1xsi32>
    %1 = "ttir.add"(%0, %arg2) : (tensor<8192x1xsi32>, tensor<8192x1xsi32>) -> tensor<8192x1xsi32>
    return %1 : tensor<8192x1xsi32>
  }
}
