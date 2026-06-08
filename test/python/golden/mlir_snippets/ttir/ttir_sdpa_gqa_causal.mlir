// GQA + causal SDPA pipeclean snippet.
//
// Shapes (tile-aligned, small):
//   Hq=8 query heads, Hkv=2 kv heads -> groups=4 GQA grouping.
//   Sq=Sk=64 (causal requires Sq==Sk).
//   D=64 head size.
//
// Decomposition path:
//   ttir-decomposition lowers ttir.scaled_dot_product_attention to:
//     reshape (GQA) -> permute (K^T) -> matmul -> reshape -> multiply (scale)
//     -> arange/arange/ge/where/add (causal mask)
//     -> softmax {numeric_stable=true} -> reshape -> matmul -> reshape
//   ttir-decompose-composites then lowers softmax into max + sub + exp + sum + div.
module {
  func.func @sdpa_gqa_causal(
      %arg0: tensor<1x8x64x64xbf16>,
      %arg1: tensor<1x2x64x64xbf16>,
      %arg2: tensor<1x2x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    %0 = "ttir.scaled_dot_product_attention"(%arg0, %arg1, %arg2) <{
      is_causal = true,
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>
    }> : (tensor<1x8x64x64xbf16>, tensor<1x2x64x64xbf16>, tensor<1x2x64x64xbf16>)
      -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }
}
