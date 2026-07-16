// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

// Qwen3-style decode attention: a per-head RMSNorm (q_norm / k_norm) sits between
// the head split and the rotary. The fused decode attention path does not model
// that norm, so the decode RoPE/QKV fusing must NOT fire here — the chain must
// stay unfused (generic rotary_embedding, no token_index). This is the guard for
// the qwen_3/causal_lm PCC regression bisected to tt-mlir #8931; without it the
// fusing mis-computes Q/K (negative PCC). A no-norm decode chain (see
// rope_decode_folded_reshape.mlir) must still fuse — the discriminator is the
// presence of an RMSNorm in the RoPE input chain.

module {
  // CHECK-LABEL: @rope_decode_qknorm_no_decode_fuse
  // The RMSNorm survives, the RoPE stays a generic (non-decode) rotary, and no
  // decode-mode rotary (token_index) is produced.
  // CHECK: "ttnn.rms_norm"
  // CHECK-NOT: token_index
  func.func @rope_decode_qknorm_no_decode_fuse(%x: tensor<32x8x1x64xbf16>, %w: tensor<64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x32x8x64xbf16> {
    // Per-head QK-norm on the head dim (D == 64), as in Qwen3's q_norm / k_norm.
    %n = "ttir.rms_norm"(%x, %w) <{normalized_shape = array<i64: 64>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x8x1x64xbf16>, tensor<64xbf16>) -> tensor<32x8x1x64xbf16>

    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%n, %cos_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    %x_hi = "ttir.slice_static"(%n) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %x_lo = "ttir.slice_static"(%n) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 1 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %rope = "ttir.add"(%x_cos, %rot_sin) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    // Layout-preserving reshape BHSD -> SBHD (seq == 1): the canonicalized form
    // of permute [2,0,1,3] (the decode layout transform).
    %result = "ttir.reshape"(%rope) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<1x32x8x64xbf16>
    return %result : tensor<1x32x8x64xbf16>
  }
}
