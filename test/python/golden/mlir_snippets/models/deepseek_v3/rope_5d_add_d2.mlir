// DeepSeek-V3 subgraph (issue #8541, d2m_subgraph_8/16/23/31).
// 5D RoPE-style cos chain (real part):
//   (q * cos) + (q_rot * cos), broadcast on the outer batch dim.
// This subgraph triggers the crash mentioned in the issue:
//   In buildPhysicalImplicitBcastIndexingMaps we compute inShard[0] = 16 and
//   outShard[0] = 512. We expect either inShard[0] == outShard[0] (no
//   broadcast) or inShard[0] == 1 (simple broadcast), but neither holds for
//   this batched 5D layout.

module {
  func.func @rope_5d_add_d2(
      %arg0: tensor<32x16x1x32x1xf32>,
      %arg1: tensor<1x16x1x32x1xf32>,
      %arg2: tensor<32x16x1x32x1xf32>,
      %arg3: tensor<1x16x1x32x1xf32>) -> tensor<32x16x1x32x1xf32> {
    %0 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x16x1x32x1xf32>, tensor<1x16x1x32x1xf32>) -> tensor<32x16x1x32x1xf32>
    %1 = "ttir.multiply"(%arg2, %arg3) : (tensor<32x16x1x32x1xf32>, tensor<1x16x1x32x1xf32>) -> tensor<32x16x1x32x1xf32>
    %2 = "ttir.add"(%0, %1) : (tensor<32x16x1x32x1xf32>, tensor<32x16x1x32x1xf32>) -> tensor<32x16x1x32x1xf32>
    return %2 : tensor<32x16x1x32x1xf32>
  }
}
