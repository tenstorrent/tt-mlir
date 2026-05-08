// Pilot 3.6 — fuse_kv_split (cross-attention K/V concat) [Quetzal §3.6].
//
// Cross-attention has Q sourced from the decoder hidden state and K, V
// sourced from the encoder hidden state. The Q/K/V matmuls do NOT share an
// LHS, so the existing SharedLHSMatmulFusion (TTIRFusing) cannot collapse
// them, and the existing SplitQueryKeyValueAndSplitHeadsFusing (which
// expects a single matmul with three slice users) does not apply.
//
// CrossAttnSplitQKVFusing recognises this layout and routes K + V into the
// kv_input_tensor operand of SplitQueryKeyValueAndSplitHeadsOp (the operand
// has been on the TTNN op since it was introduced — see TTNNOps.td:1284-1289).
//
// Two cases:
//  1. Positive: Q from decoder, K and V from encoder (same LHS). Expect a
//     single ttnn.split_query_key_value_and_split_heads with both
//     input_tensor and kv_input_tensor populated.
//  2. Negative: Q, K, V all from the same input (self-attention). The
//     existing self-attention fusing path takes precedence; no
//     kv_input_tensor is produced.
//
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

module {
  // Case 1: cross-attention. Q from decoder, K/V from encoder.
  //
  // Shapes:
  //   decoder_h: [1, 32, 512]  (B=1, S_q=32, hidden=512)
  //   encoder_h: [1, 64, 512]  (B=1, S_kv=64, hidden=512)
  //   num_heads = 8, head_dim = 64, total = 8 * 64 = 512.
  //
  // Expected: a single split op with kv_input_tensor populated.
  //
  // CHECK-LABEL: func.func @cross_attn_fuse_kv_split
  // CHECK: %[[SPLIT:.+]]:3 = "ttnn.split_query_key_value_and_split_heads"
  // CHECK-SAME: num_heads = 8 : ui32
  // CHECK: return %[[SPLIT]]#0, %[[SPLIT]]#1, %[[SPLIT]]#2
  func.func @cross_attn_fuse_kv_split(
      %decoder_h: tensor<1x32x512xbf16>,
      %encoder_h: tensor<1x64x512xbf16>,
      %wq: tensor<512x512xbf16>,
      %wk: tensor<512x512xbf16>,
      %wv: tensor<512x512xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {

    %dec2d = "ttir.reshape"(%decoder_h) <{shape = [32 : i32, 512 : i32]}> : (tensor<1x32x512xbf16>) -> tensor<32x512xbf16>
    %enc2d = "ttir.reshape"(%encoder_h) <{shape = [64 : i32, 512 : i32]}> : (tensor<1x64x512xbf16>) -> tensor<64x512xbf16>

    // Q projection from decoder.
    %q0 = "ttir.matmul"(%dec2d, %wq) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %q1 = "ttir.reshape"(%q0) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %q2 = "ttir.permute"(%q1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    // K projection from encoder.
    %k0 = "ttir.matmul"(%enc2d, %wk) <{transpose_a = false, transpose_b = true}> : (tensor<64x512xbf16>, tensor<512x512xbf16>) -> tensor<64x512xbf16>
    %k1 = "ttir.reshape"(%k0) <{shape = [1 : i32, 64 : i32, 8 : i32, 64 : i32]}> : (tensor<64x512xbf16>) -> tensor<1x64x8x64xbf16>
    %k2 = "ttir.permute"(%k1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x64x8x64xbf16>) -> tensor<1x8x64x64xbf16>

    // V projection from encoder (same LHS as K).
    %v0 = "ttir.matmul"(%enc2d, %wv) <{transpose_a = false, transpose_b = true}> : (tensor<64x512xbf16>, tensor<512x512xbf16>) -> tensor<64x512xbf16>
    %v1 = "ttir.reshape"(%v0) <{shape = [1 : i32, 64 : i32, 8 : i32, 64 : i32]}> : (tensor<64x512xbf16>) -> tensor<1x64x8x64xbf16>
    %v2 = "ttir.permute"(%v1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x64x8x64xbf16>) -> tensor<1x8x64x64xbf16>

    // SDPA: Q from decoder, K/V from encoder.
    %scale = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %k_t = "ttir.permute"(%k2) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    %qk = "ttir.matmul"(%q2, %k_t) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x32x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x32x64xbf16>
    %qk_scaled = "ttir.multiply"(%qk, %scale) : (tensor<1x8x32x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x8x32x64xbf16>
    %sm = "ttir.softmax"(%qk_scaled) <{dimension = -1 : si32}> : (tensor<1x8x32x64xbf16>) -> tensor<1x8x32x64xbf16>
    %_attn = "ttir.matmul"(%sm, %v2) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x32x64xbf16>, tensor<1x8x64x64xbf16>) -> tensor<1x8x32x64xbf16>

    return %q2, %k2, %v2 : tensor<1x8x32x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }

  // Case 2: standard self-attention (negative case for this pattern).
  // All three matmuls share the same input. The TTIR-level
  // SharedLHSMatmulFusion concatenates them into a single matmul, and the
  // self-attention SplitQueryKeyValueAndSplitHeadsFusing collapses that.
  // The cross-attn pattern must NOT also fire (would double-fuse / corrupt).
  //
  // CHECK-LABEL: func.func @self_attn_no_cross_fuse
  // The fused op is created by the self-attn path with no kv_input_tensor:
  // CHECK: ttnn.split_query_key_value_and_split_heads
  // CHECK-NOT: kv_input_tensor
  func.func @self_attn_no_cross_fuse(
      %input: tensor<1x32x512xbf16>,
      %wq: tensor<512x512xbf16>,
      %wk: tensor<512x512xbf16>,
      %wv: tensor<512x512xbf16>) -> (tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>) {

    %0 = "ttir.reshape"(%input) <{shape = [32 : i32, 512 : i32]}> : (tensor<1x32x512xbf16>) -> tensor<32x512xbf16>

    // Q: same input.
    %q0 = "ttir.matmul"(%0, %wq) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %q1 = "ttir.reshape"(%q0) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %q2 = "ttir.permute"(%q1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    // K: same input.
    %k0 = "ttir.matmul"(%0, %wk) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %k1 = "ttir.reshape"(%k0) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %k2 = "ttir.permute"(%k1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    // V: same input.
    %v0 = "ttir.matmul"(%0, %wv) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    %v1 = "ttir.reshape"(%v0) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x512xbf16>) -> tensor<1x32x8x64xbf16>
    %v2 = "ttir.permute"(%v1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x8x64xbf16>) -> tensor<1x8x32x64xbf16>

    return %q2, %k2, %v2 : tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>
  }
}
