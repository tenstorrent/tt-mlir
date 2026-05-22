// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_7/22).
// Integer index combine: ((a*b) + (c*d)) + e on si32 with scalar broadcasts.

module {
  func.func @mul_mul_add_add_si32(
      %arg0: tensor<32768x1xsi32>,
      %arg1: tensor<1x1xsi32>,
      %arg2: tensor<32768x1xsi32>,
      %arg3: tensor<1x1xsi32>,
      %arg4: tensor<32768x1xsi32>) -> tensor<32768x1xsi32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<32768x1xsi32>, tensor<1x1xsi32>) -> tensor<32768x1xsi32>
    %1 = "ttir.multiply"(%arg2, %arg3) : (tensor<32768x1xsi32>, tensor<1x1xsi32>) -> tensor<32768x1xsi32>
    %2 = "ttir.add"(%0, %1) : (tensor<32768x1xsi32>, tensor<32768x1xsi32>) -> tensor<32768x1xsi32>
    %3 = "ttir.add"(%2, %arg4) : (tensor<32768x1xsi32>, tensor<32768x1xsi32>) -> tensor<32768x1xsi32>
    return %3 : tensor<32768x1xsi32>
  }
}
