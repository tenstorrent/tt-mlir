// NLP Concat Heads Decode fusing tests.
//
// Pattern: permute([1, 2, 0, 3]) + reshape on [S=1, B, H, D] tensors.
//
// Tests cover:
//   1. Basic decode pattern: permute + reshape fuses to nlp_concat_heads_decode
//   2. Negative: wrong permutation (should not fuse)
//   3. Negative: non-decode seq_len > 1 (should not fuse)

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true" %s | FileCheck %s

module {

  // Basic decode concat heads: permute [1,2,0,3] on [1,32,8,128] + reshape.
  // CHECK-LABEL: @nlp_concat_heads_decode_basic
  // CHECK: "ttnn.nlp_concat_heads_decode"
  // CHECK-NOT: "ttnn.permute"
  func.func @nlp_concat_heads_decode_basic(%arg0: tensor<1x32x8x128xbf16>) -> tensor<32x1024xbf16> {
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 0, 3>}> : (tensor<1x32x8x128xbf16>) -> tensor<32x8x1x128xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [32 : i32, 1024 : i32]}> : (tensor<32x8x1x128xbf16>) -> tensor<32x1024xbf16>
    return %1 : tensor<32x1024xbf16>
  }

  // Negative: wrong permutation [0, 2, 1, 3] should not fuse.
  // CHECK-LABEL: @nlp_concat_heads_wrong_permutation
  // CHECK-NOT: "ttnn.nlp_concat_heads_decode"
  func.func @nlp_concat_heads_wrong_permutation(%arg0: tensor<1x32x8x128xbf16>) -> tensor<1x8x32x128xbf16> {
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x128xbf16>) -> tensor<1x8x32x128xbf16>
    return %0 : tensor<1x8x32x128xbf16>
  }

  // Negative: seq_len != 1 should not fuse.
  // CHECK-LABEL: @nlp_concat_heads_not_decode
  // CHECK-NOT: "ttnn.nlp_concat_heads_decode"
  func.func @nlp_concat_heads_not_decode(%arg0: tensor<128x32x8x64xbf16>) -> tensor<32x8x128x64xbf16> {
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 0, 3>}> : (tensor<128x32x8x64xbf16>) -> tensor<32x8x128x64xbf16>
    return %0 : tensor<32x8x128x64xbf16>
  }
}
