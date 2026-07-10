// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" --mlir-print-ir-after=ttnn-fusing %models/single_blocks_and_layers/qwen_2_5_0_5b_decode_layer.mlir 2>&1 | FileCheck %s

// Decode attention fusing on a real single-layer model IR.
//
// The decode head-split chain (RoPEDecodeFusing -> SplitQueryKeyValueAndSplitHeads
// -> NLPCreateQKVHeadsDecode) is anchored on the layout-preserving permute
// [2,0,1,3] (BHSD -> SBHD, seq == 1). The canonicalizer that runs before
// ttnn-fusing folds that permute into the adjacent reshape, so the fusing
// patterns must recognize the folded reshape form as well.
//
// Beyond just producing the decode op, this test guards the Q/K/V wiring: the
// fused op splits its input positionally into (query, key, value), and RoPE
// must land on the *key* output (rotated) that feeds the key cache, while the
// *value* output must reach the value cache un-rotated. A K/V role mix-up
// (see issue #8958) rotates the value output and swaps the caches. The IR is
// checked right after ttnn-fusing, before layout passes add memory-config ops.

// CHECK: %query, %key, %value = "ttnn.nlp_create_qkv_heads_decode"
// The key output is rotated (RoPE) and written to the key cache.
// CHECK: %[[ROPE_K:[0-9]+]] = "ttnn.rotary_embedding"(%key,
// CHECK: "ttnn.paged_update_cache"(%[[KCACHE:arg[0-9]+]], %[[ROPE_K]],
// The value output is written to the value cache directly (no RoPE).
// CHECK: "ttnn.paged_update_cache"(%[[VCACHE:arg[0-9]+]], %value,
// The query output is rotated and attention reads key cache then value cache.
// CHECK: %[[ROPE_Q:[0-9]+]] = "ttnn.rotary_embedding"(%query,
// CHECK: "ttnn.scaled_dot_product_attention_decode"(%[[ROPE_Q]], %[[KCACHE]], %[[VCACHE]],
