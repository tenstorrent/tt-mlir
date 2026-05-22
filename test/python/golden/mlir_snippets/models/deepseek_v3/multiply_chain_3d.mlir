// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_0/5).
// 2 multiplies with implicit broadcast: (32x16x896 * 32x16x1) * 1x1x896.

module {
  func.func @multiply_chain_3d(
      %arg0: tensor<32x16x896xf32>,
      %arg1: tensor<32x16x1xf32>,
      %arg2: tensor<1x1x896xf32>) -> tensor<32x16x896xf32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x16x896xf32>, tensor<32x16x1xf32>) -> tensor<32x16x896xf32>
    %1 = "ttir.multiply"(%arg2, %0) : (tensor<1x1x896xf32>, tensor<32x16x896xf32>) -> tensor<32x16x896xf32>
    return %1 : tensor<32x16x896xf32>
  }
}
