// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

// Qwen3-style decode attention: a per-head RMSNorm (q_norm / k_norm) sits between
// the head split and the rotary. The decode RoPE rewrite permutes the RoPE input
// into decode layout. When that RMSNorm is applied *directly* to the RoPE input,
// is *single-use*, and the decode permute *preserves the normalized last dim*,
// the norm is folded through the pre-permute (permute the norm's input, re-emit
// the norm on the permuted tensor) so the decode RoPE fuses while staying
// numerically correct -- recovering the perf that #9042's blanket bail cost
// (tt-mlir #9110). Any not-proven-safe placement (norm not single-use, behind
// other transforms, distributed, or a perm that moves the last dim) must still
// bail and stay unfused -- the guard for the qwen_3/causal_lm PCC regression
// bisected to tt-mlir #8931.

module {
  // A direct, single-use QK-norm folds through the decode pre-permute: the
  // RMSNorm survives (re-emitted on the permuted input) and the RoPE becomes a
  // decode-mode rotary (token_index present).
  // CHECK-LABEL: @rope_decode_qknorm_folds_through_permute
  // CHECK: "ttnn.rms_norm"
  // CHECK: token_index
  func.func @rope_decode_qknorm_folds_through_permute(%x: tensor<32x8x1x64xbf16>, %w: tensor<64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x32x8x64xbf16> {
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
    // of permute [2,0,1,3] (the decode layout transform, last dim preserved).
    %result = "ttir.reshape"(%rope) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<1x32x8x64xbf16>
    return %result : tensor<1x32x8x64xbf16>
  }

  // Same decode chain, but the RMSNorm result is *also* consumed outside the RoPE
  // (returned), so it is not single-use. Folding would permute the norm's input,
  // changing the value the other consumer sees -- so the fold must NOT fire. The
  // chain stays unfused (generic rotary, no token_index). This exercises the
  // hasOneUse() guard directly.
  // CHECK-LABEL: @rope_decode_qknorm_multi_use_no_fuse
  // CHECK: "ttnn.rms_norm"
  // CHECK-NOT: token_index
  func.func @rope_decode_qknorm_multi_use_no_fuse(%x: tensor<32x8x1x64xbf16>, %w: tensor<64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> (tensor<1x32x8x64xbf16>, tensor<32x8x1x64xbf16>) {
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

    %result = "ttir.reshape"(%rope) <{shape = [1 : i32, 32 : i32, 8 : i32, 64 : i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<1x32x8x64xbf16>
    // Second consumer of %n keeps the norm multi-use.
    return %result, %n : tensor<1x32x8x64xbf16>, tensor<32x8x1x64xbf16>
  }
}
