// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_9/17/24/32).
// 5D RoPE-style sin chain (imag part):
//   (q * cos) - (q_rot * sin), broadcast on the outer batch dim.

module {
  func.func @rope_5d_sub_d2(
      %arg0: tensor<32x16x1x32x1xf32>,
      %arg1: tensor<1x16x1x32x1xf32>,
      %arg2: tensor<32x16x1x32x1xf32>,
      %arg3: tensor<1x16x1x32x1xf32>) -> tensor<32x16x1x32x1xf32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x16x1x32x1xf32>, tensor<1x16x1x32x1xf32>) -> tensor<32x16x1x32x1xf32>
    %1 = "ttir.multiply"(%arg2, %arg3) : (tensor<32x16x1x32x1xf32>, tensor<1x16x1x32x1xf32>) -> tensor<32x16x1x32x1xf32>
    %2 = "ttir.subtract"(%0, %1) : (tensor<32x16x1x32x1xf32>, tensor<32x16x1x32x1xf32>) -> tensor<32x16x1x32x1xf32>
    return %2 : tensor<32x16x1x32x1xf32>
  }
}
