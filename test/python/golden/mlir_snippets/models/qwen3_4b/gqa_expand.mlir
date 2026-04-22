// GQA (Grouped Query Attention) key expansion from Qwen3 4B: 8 KV heads -> 32 heads
// reshape -> broadcast -> reshape -> typecast -> permute

module {
  func.func @gqa_expand(%arg0: tensor<32x8x18x128xbf16>) -> tensor<32x32x128x18xf32> {
    %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 8 : i32, 1 : i32, 18 : i32, 128 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<32x8x1x18x128xbf16>
    %1 = "ttir.broadcast"(%0) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x18x128xbf16>) -> tensor<32x8x4x18x128xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [32 : i32, 32 : i32, 18 : i32, 128 : i32]}> : (tensor<32x8x4x18x128xbf16>) -> tensor<32x32x18x128xbf16>
    %3 = "ttir.typecast"(%2) <{conservative_folding = false}> : (tensor<32x32x18x128xbf16>) -> tensor<32x32x18x128xf32>
    %4 = "ttir.permute"(%3) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<32x32x18x128xf32>) -> tensor<32x32x128x18xf32>
    return %4 : tensor<32x32x128x18xf32>
  }
}
