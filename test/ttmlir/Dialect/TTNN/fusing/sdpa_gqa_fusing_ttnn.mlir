// RUN: ttmlir-opt --ttcore-register-device --ttnn-fusing="enable-sdpa-gqa-fusion=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// RUN: ttmlir-opt --ttcore-register-device --ttnn-fusing -o %t.off %s
// RUN: FileCheck %s --check-prefix=OFF --input-file=%t.off

// These tests are written in the TTNN dialect because they exercise a pattern
// that only appears post-lowering: repeat_interleave (repeat_kv) feeding
// scaled_dot_product_attention in a grouped-query-attention model.

#dram = #ttnn.buffer_type<dram>
// Q and the *expanded* K/V: [1, 32, 128, 64].
#ttnn_layout_q = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 128 + d2, d3), <1x1>, memref<128x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Un-expanded K/V: [1, 8, 128, 64].
#ttnn_layout_kv = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Attention mask: [1, 1, 128, 128].
#ttnn_layout_mask = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // GQA: 32 query heads, 8 KV heads (group size 4). The repeat_interleave
  // expansion of K and V should be removed and the un-expanded [1,8,128,64]
  // K/V fed straight to SDPA.

  // CHECK-LABEL: func.func @sdpa_gqa
  // OFF-LABEL:   func.func @sdpa_gqa
  func.func @sdpa_gqa(%q: tensor<1x32x128x64xbf16, #ttnn_layout_q>,
                      %k: tensor<1x8x128x64xbf16, #ttnn_layout_kv>,
                      %v: tensor<1x8x128x64xbf16, #ttnn_layout_kv>,
                      %mask: tensor<1x1x128x128xbf16, #ttnn_layout_mask>)
                      -> tensor<1x32x128x64xbf16, #ttnn_layout_q> {
    // CHECK-NOT: ttnn.repeat_interleave
    // OFF: ttnn.repeat_interleave
    %ke = "ttnn.repeat_interleave"(%k) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x8x128x64xbf16, #ttnn_layout_kv>) -> tensor<1x32x128x64xbf16, #ttnn_layout_q>
    %ve = "ttnn.repeat_interleave"(%v) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x8x128x64xbf16, #ttnn_layout_kv>) -> tensor<1x32x128x64xbf16, #ttnn_layout_q>

    // After fusing, key and value are the un-expanded [1,8,128,64] tensors.
    // CHECK: "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3)
    // CHECK-SAME: tensor<1x32x128x64xbf16{{.*}}>, tensor<1x8x128x64xbf16{{.*}}>, tensor<1x8x128x64xbf16{{.*}}>, tensor<1x1x128x128xbf16{{.*}}>
    // OFF: "ttnn.scaled_dot_product_attention"
    %out = "ttnn.scaled_dot_product_attention"(%q, %ke, %ve, %mask) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, scale = 0.0883883461 : f32}> : (tensor<1x32x128x64xbf16, #ttnn_layout_q>, tensor<1x32x128x64xbf16, #ttnn_layout_q>, tensor<1x32x128x64xbf16, #ttnn_layout_q>, tensor<1x1x128x128xbf16, #ttnn_layout_mask>) -> tensor<1x32x128x64xbf16, #ttnn_layout_q>
    return %out : tensor<1x32x128x64xbf16, #ttnn_layout_q>
  }

  // Negative: repeat_interleave on a non-head dimension (dim=2) must NOT be
  // folded, since it does not correspond to KV-head broadcast.
  // CHECK-LABEL: func.func @sdpa_repeat_wrong_dim
  func.func @sdpa_repeat_wrong_dim(%q: tensor<1x32x128x64xbf16, #ttnn_layout_q>,
                                   %k: tensor<1x32x32x64xbf16, #ttnn_layout_q>,
                                   %v: tensor<1x32x32x64xbf16, #ttnn_layout_q>,
                                   %mask: tensor<1x1x128x128xbf16, #ttnn_layout_mask>)
                                   -> tensor<1x32x128x64xbf16, #ttnn_layout_q> {
    // Repeat is on the sequence dim, not the head dim -> not fusable.
    // CHECK: ttnn.repeat_interleave
    %ke = "ttnn.repeat_interleave"(%k) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x32x32x64xbf16, #ttnn_layout_q>) -> tensor<1x32x128x64xbf16, #ttnn_layout_q>
    %ve = "ttnn.repeat_interleave"(%v) <{dim = 2 : si32, repeats = 4 : ui32}> : (tensor<1x32x32x64xbf16, #ttnn_layout_q>) -> tensor<1x32x128x64xbf16, #ttnn_layout_q>
    %out = "ttnn.scaled_dot_product_attention"(%q, %ke, %ve, %mask) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>, scale = 0.0883883461 : f32}> : (tensor<1x32x128x64xbf16, #ttnn_layout_q>, tensor<1x32x128x64xbf16, #ttnn_layout_q>, tensor<1x32x128x64xbf16, #ttnn_layout_q>, tensor<1x1x128x128xbf16, #ttnn_layout_mask>) -> tensor<1x32x128x64xbf16, #ttnn_layout_q>
    return %out : tensor<1x32x128x64xbf16, #ttnn_layout_q>
  }
}
