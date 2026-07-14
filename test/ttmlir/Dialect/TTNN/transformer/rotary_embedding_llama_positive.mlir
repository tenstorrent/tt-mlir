// RUN: ttmlir-opt %s | FileCheck %s

// In decode mode the transformation matrix is height-sharded across the batch,
// so its logical shape is (1, 1, N*32, 32) rather than a single (1, 1, 32, 32)
// tile. The verifier must accept the batched shape (tt-metal does not pin the
// matrix to a single tile).
#dram = #ttnn.buffer_type<dram>
#enc = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#enc_tm = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @rope_batched_decode_trans_mat(%input: tensor<1x1x32x32xbf16, #enc>, %cos: tensor<1x1x32x32xbf16, #enc>, %sin: tensor<1x1x32x32xbf16, #enc>, %trans_mat: tensor<1x1x64x32xbf16, #enc_tm>) -> tensor<1x1x32x32xbf16, #enc> {
    // CHECK: "ttnn.rotary_embedding_llama"
    %0 = "ttnn.rotary_embedding_llama"(%input, %cos, %sin, %trans_mat) <{ is_decode_mode = true }> : (tensor<1x1x32x32xbf16, #enc>, tensor<1x1x32x32xbf16, #enc>, tensor<1x1x32x32xbf16, #enc>, tensor<1x1x64x32xbf16, #enc_tm>) -> tensor<1x1x32x32xbf16, #enc>
    return %0 : tensor<1x1x32x32xbf16, #enc>
  }
}
