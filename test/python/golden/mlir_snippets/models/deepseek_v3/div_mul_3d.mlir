// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_3).
// 3D normalization: (a / b) * c with column/scalar broadcasts.

module {
  func.func @div_mul_3d(
      %arg0: tensor<512x1x8xf32>,
      %arg1: tensor<512x1x1xf32>,
      %arg2: tensor<1x1x1xf32>) -> tensor<512x1x8xf32> {
    %0 = "ttir.div"(%arg0, %arg1) : (tensor<512x1x8xf32>, tensor<512x1x1xf32>) -> tensor<512x1x8xf32>
    %1 = "ttir.multiply"(%0, %arg2) : (tensor<512x1x8xf32>, tensor<1x1x1xf32>) -> tensor<512x1x8xf32>
    return %1 : tensor<512x1x8xf32>
  }
}
