// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_10/25).
// 5D RoPE-style cos chain with 16x context dim:
//   (q * cos) + (q_rot * cos), broadcast on the outer batch and middle dims.

module {
  func.func @rope_5d_add_d16(
      %arg0: tensor<32x16x16x32x1xf32>,
      %arg1: tensor<1x16x1x32x1xf32>,
      %arg2: tensor<32x16x16x32x1xf32>,
      %arg3: tensor<1x16x1x32x1xf32>) -> tensor<32x16x16x32x1xf32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x16x16x32x1xf32>, tensor<1x16x1x32x1xf32>) -> tensor<32x16x16x32x1xf32>
    %1 = "ttir.multiply"(%arg2, %arg3) : (tensor<32x16x16x32x1xf32>, tensor<1x16x1x32x1xf32>) -> tensor<32x16x16x32x1xf32>
    %2 = "ttir.add"(%0, %1) : (tensor<32x16x16x32x1xf32>, tensor<32x16x16x32x1xf32>) -> tensor<32x16x16x32x1xf32>
    return %2 : tensor<32x16x16x32x1xf32>
  }
}
