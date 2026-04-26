// RoPE (Rotary Position Embedding) fusing tests.
//
// Pattern: (x * cos) + (rotate_half(x) * sin)
//   where rotate_half(x) = concat(neg(slice(x, half:)), slice(x, :half))
//
// Tests cover:
//   1. Basic BHSD layout with broadcast cos/sin (3D → 4D reshape + broadcast)
//   2. 4D cos/sin with broadcast over heads
//   3. Commuted operand order (sin branch on lhs of add)
//   4. Commuted multiply operands
//   5. Typecast in TM chain
//   6. Decode mode (seq_len=1)

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

module {

  // Basic RoPE: 3D cos/sin reshaped and broadcast to match x [1,32,1024,64].
  // CHECK-LABEL: @rope_basic_broadcast
  // CHECK: ttnn.rotary_embedding"
  func.func @rope_basic_broadcast(%sin: tensor<1x1024x64xbf16>, %x: tensor<1x32x1024x64xbf16>, %cos: tensor<1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16> {
    %cos4d = "ttir.reshape"(%cos) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    %cos_bc = "ttir.broadcast"(%cos4d) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 32:i32, 1024:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x32x1024x32xbf16>) -> tensor<1x32x1024x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 32:i32, 1024:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x32x1024x32xbf16>, tensor<1x32x1024x32xbf16>) -> tensor<1x32x1024x64xbf16>

    %sin4d = "ttir.reshape"(%sin) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    %sin_bc = "ttir.broadcast"(%sin4d) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>

    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x32x1024x64xbf16>, tensor<1x32x1024x64xbf16>) -> tensor<1x32x1024x64xbf16>
    return %result : tensor<1x32x1024x64xbf16>
  }

  // 4D cos/sin [1,1,S,D] broadcast over heads before multiply.
  // CHECK-LABEL: @rope_4d_broadcast
  // CHECK: ttnn.rotary_embedding"
  func.func @rope_4d_broadcast(%x: tensor<1x32x128x64xbf16>, %cos: tensor<1x1x128x64xbf16>, %sin: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 32:i32, 128:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x32x128x32xbf16>) -> tensor<1x32x128x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 32:i32, 128:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x32x128x32xbf16>, tensor<1x32x128x32xbf16>) -> tensor<1x32x128x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %result : tensor<1x32x128x64xbf16>
  }

  // Commuted add operands: sin branch on LHS, cos branch on RHS.
  // CHECK-LABEL: @rope_commuted_add
  // CHECK: ttnn.rotary_embedding"
  func.func @rope_commuted_add(%x: tensor<1x32x128x64xbf16>, %cos: tensor<1x1x128x64xbf16>, %sin: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 32:i32, 128:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x32x128x32xbf16>) -> tensor<1x32x128x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 32:i32, 128:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x32x128x32xbf16>, tensor<1x32x128x32xbf16>) -> tensor<1x32x128x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    // Note: sin branch (rot_sin) is LHS, cos branch (x_cos) is RHS
    %result = "ttir.add"(%rot_sin, %x_cos) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %result : tensor<1x32x128x64xbf16>
  }

  // Commuted multiply operands: cos * x and sin * rotated.
  // CHECK-LABEL: @rope_commuted_multiply
  // CHECK: ttnn.rotary_embedding"
  func.func @rope_commuted_multiply(%x: tensor<1x32x128x64xbf16>, %cos: tensor<1x1x128x64xbf16>, %sin: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %x_cos = "ttir.multiply"(%cos_bc, %x) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 32:i32, 128:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x32x128x32xbf16>) -> tensor<1x32x128x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 32:i32, 128:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x32x128x32xbf16>, tensor<1x32x128x32xbf16>) -> tensor<1x32x128x64xbf16>

    // Note: sin_bc is LHS operand, rotated is RHS
    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %rot_sin = "ttir.multiply"(%sin_bc, %rotated) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %result : tensor<1x32x128x64xbf16>
  }

  // Typecast in the TM chain before the RoPE computation.
  // CHECK-LABEL: @rope_with_typecast
  // CHECK: ttnn.rotary_embedding"
  func.func @rope_with_typecast(%x_bf16: tensor<1x32x128x64xbf16>, %cos: tensor<1x1x128x64xbf16>, %sin: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    %x = "ttir.typecast"(%x_bf16) <{conservative_folding = false}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 32:i32, 128:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x32x128x32xbf16>) -> tensor<1x32x128x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 32:i32, 128:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x32x128x32xbf16>, tensor<1x32x128x32xbf16>) -> tensor<1x32x128x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %result : tensor<1x32x128x64xbf16>
  }

  // Decode shape: seq_len=1, head_dim=64.
  // CHECK-LABEL: @rope_decode
  // CHECK: ttnn.rotary_embedding"
  func.func @rope_decode(%x: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x8x1x32xbf16>, tensor<1x8x1x32xbf16>) -> tensor<1x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }

  // =========================================================================
  // Negative tests: patterns that look like RoPE but must NOT fuse.
  // =========================================================================

  // Both halves negated: neg(hi) concat neg(lo) — not a valid rotate_half.
  // CHECK-LABEL: @rope_neg_both_halves_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.add"
  func.func @rope_neg_both_halves_no_fuse(%x: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    // Both halves negated — invalid rotate_half.
    %neg_lo = "ttir.neg"(%x_lo) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %neg_lo) <{dim = 3 : si32}> : (tensor<1x8x1x32xbf16>, tensor<1x8x1x32xbf16>) -> tensor<1x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }

  // Non-complementary slices: both slices take the first half [0:32].
  // CHECK-LABEL: @rope_same_half_slices_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.add"
  func.func @rope_same_half_slices_no_fuse(%x: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    // Both slices take [0:32] — not complementary halves.
    %x_lo1 = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %neg_lo1 = "ttir.neg"(%x_lo1) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %x_lo2 = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_lo1, %x_lo2) <{dim = 3 : si32}> : (tensor<1x8x1x32xbf16>, tensor<1x8x1x32xbf16>) -> tensor<1x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }

  // Different x sources: cos branch uses x, sin branch uses y.
  // CHECK-LABEL: @rope_different_x_sources_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.add"
  func.func @rope_different_x_sources_no_fuse(%x: tensor<1x8x1x64xbf16>, %y: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    // cos branch uses x
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    // sin branch uses y — different source, not valid RoPE.
    %y_hi = "ttir.slice_static"(%y) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%y_hi) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %y_lo = "ttir.slice_static"(%y) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %y_lo) <{dim = 3 : si32}> : (tensor<1x8x1x32xbf16>, tensor<1x8x1x32xbf16>) -> tensor<1x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }

  // Negated half is the wrong one: neg(first_half) instead of neg(second_half).
  // rotate_half requires concat(neg(x[D/2:]), x[:D/2]), not concat(neg(x[:D/2]), x[D/2:]).
  // CHECK-LABEL: @rope_wrong_neg_half_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.add"
  func.func @rope_wrong_neg_half_no_fuse(%x: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    // neg on first half [0:32] instead of second half [32:64].
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %neg_lo = "ttir.neg"(%x_lo) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_lo, %x_hi) <{dim = 3 : si32}> : (tensor<1x8x1x32xbf16>, tensor<1x8x1x32xbf16>) -> tensor<1x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }

  // No rotate_half at all: sin branch multiplies x directly (not rotated).
  // This is x*cos + x*sin, not RoPE.
  // CHECK-LABEL: @rope_no_rotate_half_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.add"
  func.func @rope_no_rotate_half_no_fuse(%x: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    // sin branch uses x directly — no rotation.
    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_sin = "ttir.multiply"(%x, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %x_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }

  // Permute [0,2,1,3] inserted between sin multiply and add — the sin branch
  // arrives at the add in BHSD order but swapped to BSHD, while the cos branch
  // stays BHSD. The axis analyzer should fail because the two add operands
  // have incompatible axis layouts.
  // CHECK-LABEL: @rope_permute_before_add_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.add"
  func.func @rope_permute_before_add_no_fuse(%x: tensor<1x32x32x64xbf16>, %cos: tensor<1x1x32x64xbf16>, %sin: tensor<1x1x32x64xbf16>) -> tensor<1x32x32x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x32x32x64xbf16>, tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 32:i32, 32:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 32:i32, 32:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x32x32x32xbf16>, tensor<1x32x32x32xbf16>) -> tensor<1x32x32x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x32x32x64xbf16>, tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    // Permute sin branch BHSD -> BSHD before add — breaks axis consistency.
    %rot_sin_perm = "ttir.permute"(%rot_sin) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin_perm) : (tensor<1x32x32x64xbf16>, tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    return %result : tensor<1x32x32x64xbf16>
  }

  // =========================================================================
  // Expanded (trig-identity) RoPE tests
  // =========================================================================
  //
  // Pattern: concat(
  //            subtract(x[:half] * cos_h, x[half:] * sin_h),
  //            add(x[half:] * cos_h, x[:half] * sin_h))
  // where cos_h and sin_h are half-dim embeddings.

  // Basic expanded RoPE: GPT-OSS 20B decode shapes with broadcast cos/sin.
  // cos_h/sin_h are [1,1,1,32] broadcast to [16,8,1,32] before multiply,
  // matching the real model where cos/sin are computed once and broadcast.
  // CHECK-LABEL: @rope_expanded_basic
  // CHECK: "ttnn.rotary_embedding"
  // CHECK-NOT: ttnn.subtract"
  func.func @rope_expanded_basic(
      %x: tensor<16x8x1x64xbf16>,
      %cos_h: tensor<1x1x1x32xbf16>,
      %sin_h: tensor<1x1x1x32xbf16>) -> tensor<16x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos_h) <{broadcast_dimensions = array<i64: 16, 8, 1, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %sin_bc = "ttir.broadcast"(%sin_h) <{broadcast_dimensions = array<i64: 16, 8, 1, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    %first = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [16:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<16x8x1x64xbf16>) -> tensor<16x8x1x32xbf16>
    %second = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [16:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<16x8x1x64xbf16>) -> tensor<16x8x1x32xbf16>

    %fc = "ttir.multiply"(%first, %cos_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %ss = "ttir.multiply"(%second, %sin_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %sub = "ttir.subtract"(%fc, %ss) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    %sc = "ttir.multiply"(%second, %cos_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %fs = "ttir.multiply"(%first, %sin_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %add = "ttir.add"(%sc, %fs) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    %result = "ttir.concat"(%sub, %add) <{dim = 3 : si32}> : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x64xbf16>
    return %result : tensor<16x8x1x64xbf16>
  }

  // Expanded RoPE with broadcast cos/sin: batch=32.
  // CHECK-LABEL: @rope_expanded_broadcast_b32
  // CHECK: "ttnn.rotary_embedding"
  // CHECK-NOT: ttnn.subtract"
  func.func @rope_expanded_broadcast_b32(
      %x: tensor<32x8x1x64xbf16>,
      %cos_h: tensor<1x1x1x32xbf16>,
      %sin_h: tensor<1x1x1x32xbf16>) -> tensor<32x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos_h) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %sin_bc = "ttir.broadcast"(%sin_h) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<32x8x1x32xbf16>

    %first = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [32:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>
    %second = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [32:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<32x8x1x64xbf16>) -> tensor<32x8x1x32xbf16>

    %fc = "ttir.multiply"(%first, %cos_bc) : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %ss = "ttir.multiply"(%second, %sin_bc) : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %sub = "ttir.subtract"(%fc, %ss) : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>

    %sc = "ttir.multiply"(%second, %cos_bc) : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %fs = "ttir.multiply"(%first, %sin_bc) : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>
    %add = "ttir.add"(%sc, %fs) : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x32xbf16>

    %result = "ttir.concat"(%sub, %add) <{dim = 3 : si32}> : (tensor<32x8x1x32xbf16>, tensor<32x8x1x32xbf16>) -> tensor<32x8x1x64xbf16>
    return %result : tensor<32x8x1x64xbf16>
  }

  // Expanded RoPE with head_dim=128, broadcast cos/sin.
  // CHECK-LABEL: @rope_expanded_head_dim_128
  // CHECK: "ttnn.rotary_embedding"
  // CHECK-NOT: ttnn.subtract"
  func.func @rope_expanded_head_dim_128(
      %x: tensor<32x8x1x128xbf16>,
      %cos_h: tensor<1x1x1x64xbf16>,
      %sin_h: tensor<1x1x1x64xbf16>) -> tensor<32x8x1x128xbf16> {
    %cos_bc = "ttir.broadcast"(%cos_h) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %sin_bc = "ttir.broadcast"(%sin_h) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    %first = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [32:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<32x8x1x128xbf16>) -> tensor<32x8x1x64xbf16>
    %second = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 64:i32], ends = [32:i32, 8:i32, 1:i32, 128:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<32x8x1x128xbf16>) -> tensor<32x8x1x64xbf16>

    %fc = "ttir.multiply"(%first, %cos_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %ss = "ttir.multiply"(%second, %sin_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %sub = "ttir.subtract"(%fc, %ss) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    %sc = "ttir.multiply"(%second, %cos_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %fs = "ttir.multiply"(%first, %sin_bc) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>
    %add = "ttir.add"(%sc, %fs) : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x64xbf16>

    %result = "ttir.concat"(%sub, %add) <{dim = 3 : si32}> : (tensor<32x8x1x64xbf16>, tensor<32x8x1x64xbf16>) -> tensor<32x8x1x128xbf16>
    return %result : tensor<32x8x1x128xbf16>
  }

  // Negative: wrong concat order (add first, subtract second) — not valid RoPE.
  // CHECK-LABEL: @rope_expanded_wrong_concat_order_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.concat"
  func.func @rope_expanded_wrong_concat_order_no_fuse(
      %x: tensor<16x8x1x64xbf16>,
      %cos_h: tensor<1x1x1x32xbf16>,
      %sin_h: tensor<1x1x1x32xbf16>) -> tensor<16x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos_h) <{broadcast_dimensions = array<i64: 16, 8, 1, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %sin_bc = "ttir.broadcast"(%sin_h) <{broadcast_dimensions = array<i64: 16, 8, 1, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    %first = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [16:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<16x8x1x64xbf16>) -> tensor<16x8x1x32xbf16>
    %second = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [16:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<16x8x1x64xbf16>) -> tensor<16x8x1x32xbf16>

    %fc = "ttir.multiply"(%first, %cos_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %ss = "ttir.multiply"(%second, %sin_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %sub = "ttir.subtract"(%fc, %ss) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    %sc = "ttir.multiply"(%second, %cos_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %fs = "ttir.multiply"(%first, %sin_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %add = "ttir.add"(%sc, %fs) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    // Wrong order: add first, subtract second.
    %result = "ttir.concat"(%add, %sub) <{dim = 3 : si32}> : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x64xbf16>
    return %result : tensor<16x8x1x64xbf16>
  }

  // Negative: sub.lhs uses second_half instead of first_half — breaks cross pattern.
  // CHECK-LABEL: @rope_expanded_wrong_slice_roles_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.concat"
  func.func @rope_expanded_wrong_slice_roles_no_fuse(
      %x: tensor<16x8x1x64xbf16>,
      %cos_h: tensor<1x1x1x32xbf16>,
      %sin_h: tensor<1x1x1x32xbf16>) -> tensor<16x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos_h) <{broadcast_dimensions = array<i64: 16, 8, 1, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %sin_bc = "ttir.broadcast"(%sin_h) <{broadcast_dimensions = array<i64: 16, 8, 1, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    %first = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [16:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<16x8x1x64xbf16>) -> tensor<16x8x1x32xbf16>
    %second = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [16:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<16x8x1x64xbf16>) -> tensor<16x8x1x32xbf16>

    // sub.lhs uses second_half*cos instead of first_half*cos
    %sc = "ttir.multiply"(%second, %cos_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %fs = "ttir.multiply"(%first, %sin_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %sub = "ttir.subtract"(%sc, %fs) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    %fc = "ttir.multiply"(%first, %cos_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %ss = "ttir.multiply"(%second, %sin_bc) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>
    %add = "ttir.add"(%fc, %ss) : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x32xbf16>

    %result = "ttir.concat"(%sub, %add) <{dim = 3 : si32}> : (tensor<16x8x1x32xbf16>, tensor<16x8x1x32xbf16>) -> tensor<16x8x1x64xbf16>
    return %result : tensor<16x8x1x64xbf16>
  }

  // Negative: cos/sin dim -2 is 1 but result dim -2 is 8 (heads, not seq).
  // Kernel cannot broadcast dim -2 from 1; fusing would silently produce
  // wrong outputs once the seq_len padding workaround inflates the heads
  // dim to a tile multiple. Pattern observed in tt-xla vLLM TTRotaryEmbedding.
  // CHECK-LABEL: @rope_expanded_dim_minus_2_mismatch_no_fuse
  // CHECK-NOT: ttnn.rotary_embedding"
  // CHECK: ttnn.concat"
  func.func @rope_expanded_dim_minus_2_mismatch_no_fuse(
      %x: tensor<1x1x8x64xbf16>,
      %cos_h: tensor<1x1x1x32xbf16>,
      %sin_h: tensor<1x1x1x32xbf16>) -> tensor<1x1x8x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos_h) <{broadcast_dimensions = array<i64: 1, 1, 8, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<1x1x8x32xbf16>
    %sin_bc = "ttir.broadcast"(%sin_h) <{broadcast_dimensions = array<i64: 1, 1, 8, 1>}> : (tensor<1x1x1x32xbf16>) -> tensor<1x1x8x32xbf16>

    %first = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 1:i32, 8:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x1x8x64xbf16>) -> tensor<1x1x8x32xbf16>
    %second = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 1:i32, 8:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x1x8x64xbf16>) -> tensor<1x1x8x32xbf16>

    %fc = "ttir.multiply"(%first, %cos_bc) : (tensor<1x1x8x32xbf16>, tensor<1x1x8x32xbf16>) -> tensor<1x1x8x32xbf16>
    %ss = "ttir.multiply"(%second, %sin_bc) : (tensor<1x1x8x32xbf16>, tensor<1x1x8x32xbf16>) -> tensor<1x1x8x32xbf16>
    %sub = "ttir.subtract"(%fc, %ss) : (tensor<1x1x8x32xbf16>, tensor<1x1x8x32xbf16>) -> tensor<1x1x8x32xbf16>

    %sc = "ttir.multiply"(%second, %cos_bc) : (tensor<1x1x8x32xbf16>, tensor<1x1x8x32xbf16>) -> tensor<1x1x8x32xbf16>
    %fs = "ttir.multiply"(%first, %sin_bc) : (tensor<1x1x8x32xbf16>, tensor<1x1x8x32xbf16>) -> tensor<1x1x8x32xbf16>
    %add = "ttir.add"(%sc, %fs) : (tensor<1x1x8x32xbf16>, tensor<1x1x8x32xbf16>) -> tensor<1x1x8x32xbf16>

    %result = "ttir.concat"(%sub, %add) <{dim = 3 : si32}> : (tensor<1x1x8x32xbf16>, tensor<1x1x8x32xbf16>) -> tensor<1x1x8x64xbf16>
    return %result : tensor<1x1x8x64xbf16>
  }
}
