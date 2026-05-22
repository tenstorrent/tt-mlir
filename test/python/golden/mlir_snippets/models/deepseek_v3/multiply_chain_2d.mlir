// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_18/20/33).
// 2 multiplies with implicit broadcast: (512x896 * 512x1) * 1x896.

module {
  func.func @multiply_chain_2d(
      %arg0: tensor<512x896xf32>,
      %arg1: tensor<512x1xf32>,
      %arg2: tensor<1x896xf32>) -> tensor<512x896xf32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<512x896xf32>, tensor<512x1xf32>) -> tensor<512x896xf32>
    %1 = "ttir.multiply"(%arg2, %0) : (tensor<1x896xf32>, tensor<512x896xf32>) -> tensor<512x896xf32>
    return %1 : tensor<512x896xf32>
  }
}
