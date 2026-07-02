// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %models/single_blocks_and_layers/qwen_2_5_0_5b_decode_layer.mlir | FileCheck %s

// Decode attention fusing on a real single-layer model IR.
//
// The decode head-split chain (RoPEDecodeFusing -> SplitQueryKeyValueAndSplitHeads
// -> NLPCreateQKVHeadsDecode) is anchored on the layout-preserving permute
// [2,0,1,3] (BHSD -> SBHD, seq == 1). The canonicalizer that runs before
// ttnn-fusing folds that permute into the adjacent reshape, so the fusing
// patterns must recognize the folded reshape form as well. This test guards
// that the decode op is still produced from the canonicalized IR.

// CHECK: ttnn.nlp_create_qkv_heads_decode
