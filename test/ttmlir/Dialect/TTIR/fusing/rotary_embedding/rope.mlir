// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

// =========================================================================
// Pattern 1: rotate_half RoPE
// =========================================================================

// Basic rotate_half with 4D broadcast cos/sin.
// CHECK-LABEL: @rope_rotate_half_basic
// CHECK: "ttcore.composite"
// CHECK-SAME: composite_name = "rotary_embedding"
module {
  func.func @rope_rotate_half_basic(%x: tensor<1x32x128x64xbf16>, %cos: tensor<1x1x128x64xbf16>, %sin: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
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
}

// Commuted add: sin branch on LHS.
// CHECK-LABEL: @rope_rotate_half_commuted_add
// CHECK: "ttcore.composite"
module {
  func.func @rope_rotate_half_commuted_add(%x: tensor<1x32x128x64xbf16>, %cos: tensor<1x1x128x64xbf16>, %sin: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 32:i32, 128:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x32x128x32xbf16>) -> tensor<1x32x128x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 32:i32, 128:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x32x128x32xbf16>, tensor<1x32x128x32xbf16>) -> tensor<1x32x128x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    %result = "ttir.add"(%rot_sin, %x_cos) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %result : tensor<1x32x128x64xbf16>
  }
}

// Commuted multiply: cos * x instead of x * cos.
// CHECK-LABEL: @rope_rotate_half_commuted_multiply
// CHECK: "ttcore.composite"
module {
  func.func @rope_rotate_half_commuted_multiply(%x: tensor<1x32x128x64xbf16>, %cos: tensor<1x1x128x64xbf16>, %sin: tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %x_cos = "ttir.multiply"(%cos_bc, %x) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 32:i32, 128:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x32x128x32xbf16>) -> tensor<1x32x128x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 32:i32, 128:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x32x128x32xbf16>, tensor<1x32x128x32xbf16>) -> tensor<1x32x128x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %rot_sin = "ttir.multiply"(%sin_bc, %rotated) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %result : tensor<1x32x128x64xbf16>
  }
}

// 3D cos/sin with reshape + broadcast TM chain.
// CHECK-LABEL: @rope_rotate_half_3d_reshape
// CHECK: "ttcore.composite"
module {
  func.func @rope_rotate_half_3d_reshape(%x: tensor<1x32x1024x64xbf16>, %sin: tensor<1x1024x64xbf16>, %cos: tensor<1x1024x64xbf16>) -> tensor<1x32x1024x64xbf16> {
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
}

// =========================================================================
// Pattern 2: complex rotation (expanded) RoPE
// =========================================================================

// Basic expanded RoPE with half-dim cos/sin.
// CHECK-LABEL: @rope_expanded_basic
// CHECK: "ttcore.composite"
// CHECK-SAME: composite_name = "rotary_embedding"
module {
  func.func @rope_expanded_basic(%x: tensor<16x8x1x64xbf16>, %cos_h: tensor<1x1x1x32xbf16>, %sin_h: tensor<1x1x1x32xbf16>) -> tensor<16x8x1x64xbf16> {
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
}

// =========================================================================
// Negative tests
// =========================================================================

// Both halves negated — not valid rotate_half.
// CHECK-LABEL: @rope_neg_both_halves_no_fuse
// CHECK-NOT: "ttcore.composite"
// CHECK: "ttir.add"
module {
  func.func @rope_neg_both_halves_no_fuse(%x: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    %x_hi = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%x_hi) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %x_lo = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %neg_lo = "ttir.neg"(%x_lo) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %neg_lo) <{dim = 3 : si32}> : (tensor<1x8x1x32xbf16>, tensor<1x8x1x32xbf16>) -> tensor<1x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }
}

// Different x sources: cos branch uses %x, sin branch uses %y.
// CHECK-LABEL: @rope_different_x_sources_no_fuse
// CHECK-NOT: "ttcore.composite"
// CHECK: "ttir.add"
module {
  func.func @rope_different_x_sources_no_fuse(%x: tensor<1x8x1x64xbf16>, %y: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    %y_hi = "ttir.slice_static"(%y) <{begins = [0:i32, 0:i32, 0:i32, 32:i32], ends = [1:i32, 8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %neg_hi = "ttir.neg"(%y_hi) : (tensor<1x8x1x32xbf16>) -> tensor<1x8x1x32xbf16>
    %y_lo = "ttir.slice_static"(%y) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x8x1x64xbf16>) -> tensor<1x8x1x32xbf16>
    %rotated = "ttir.concat"(%neg_hi, %y_lo) <{dim = 3 : si32}> : (tensor<1x8x1x32xbf16>, tensor<1x8x1x32xbf16>) -> tensor<1x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %rot_sin = "ttir.multiply"(%rotated, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %rot_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }
}

// No rotate_half: sin branch multiplies x directly.
// CHECK-LABEL: @rope_no_rotate_half_no_fuse
// CHECK-NOT: "ttcore.composite"
// CHECK: "ttir.add"
module {
  func.func @rope_no_rotate_half_no_fuse(%x: tensor<1x8x1x64xbf16>, %cos: tensor<1x1x1x64xbf16>, %sin: tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_cos = "ttir.multiply"(%x, %cos_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>

    %sin_bc = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %x_sin = "ttir.multiply"(%x, %sin_bc) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    %result = "ttir.add"(%x_cos, %x_sin) : (tensor<1x8x1x64xbf16>, tensor<1x8x1x64xbf16>) -> tensor<1x8x1x64xbf16>
    return %result : tensor<1x8x1x64xbf16>
  }
}

// =========================================================================
// Pattern 3: interleaved-pair RoPE
// =========================================================================

// Interleaved-pair RoPE chain at TTIR level. The matcher should fold the
// entire reshape -> slice -> reshape -> broadcast -> multiply -> add ->
// reshape chain into a single ttcore.composite (with surrounding
// permutations for interleaved <-> rotate-half conversion).
// CHECK-LABEL: @rope_interleaved_pair_basic
// CHECK: "ttcore.composite"
// CHECK-SAME: composite_name = "rotary_embedding"
module {
  func.func @rope_interleaved_pair_basic(
      %x: tensor<1x4x2x8xf32>,
      %freqs: tensor<1x4x1x4x2x2xf32>) -> tensor<1x4x2x8xf32> {

    // ---- col 0 of freqs_cis 2x2 = [cos, sin] ----
    %f0_slice = "ttir.slice_static"(%freqs) <{begins = [0:i32, 0:i32, 0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 4:i32, 1:i32, 4:i32, 2:i32, 1:i32], step = [1:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x4x1x4x2x2xf32>) -> tensor<1x4x1x4x2x1xf32>
    %f0_r1 = "ttir.reshape"(%f0_slice) <{shape = [1:i32, 4:i32, 4:i32, 2:i32]}> : (tensor<1x4x1x4x2x1xf32>) -> tensor<1x4x4x2xf32>
    %f0_r2 = "ttir.reshape"(%f0_r1) <{shape = [1:i32, 4:i32, 1:i32, 4:i32, 2:i32]}> : (tensor<1x4x4x2xf32>) -> tensor<1x4x1x4x2xf32>
    %f0_bc = "ttir.broadcast"(%f0_r2) <{broadcast_dimensions = array<i64: 1, 1, 2, 1, 1>}> : (tensor<1x4x1x4x2xf32>) -> tensor<1x4x2x4x2xf32>

    // ---- col 1 of freqs_cis 2x2 = [-sin, cos] ----
    %f1_slice = "ttir.slice_static"(%freqs) <{begins = [0:i32, 0:i32, 0:i32, 0:i32, 0:i32, 1:i32], ends = [1:i32, 4:i32, 1:i32, 4:i32, 2:i32, 2:i32], step = [1:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x4x1x4x2x2xf32>) -> tensor<1x4x1x4x2x1xf32>
    %f1_r1 = "ttir.reshape"(%f1_slice) <{shape = [1:i32, 4:i32, 4:i32, 2:i32]}> : (tensor<1x4x1x4x2x1xf32>) -> tensor<1x4x4x2xf32>
    %f1_r2 = "ttir.reshape"(%f1_r1) <{shape = [1:i32, 4:i32, 1:i32, 4:i32, 2:i32]}> : (tensor<1x4x4x2xf32>) -> tensor<1x4x1x4x2xf32>
    %f1_bc = "ttir.broadcast"(%f1_r2) <{broadcast_dimensions = array<i64: 1, 1, 2, 1, 1>}> : (tensor<1x4x1x4x2xf32>) -> tensor<1x4x2x4x2xf32>

    // ---- reshape x to expose pairs: (1,4,2,8) -> (1,4,2,4,1,2) ----
    %x_6d = "ttir.reshape"(%x) <{shape = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32, 2:i32]}> : (tensor<1x4x2x8xf32>) -> tensor<1x4x2x4x1x2xf32>

    // ---- pair index 0 (real) ----
    %x0_slice = "ttir.slice_static"(%x_6d) <{begins = [0:i32, 0:i32, 0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32, 1:i32], step = [1:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x4x2x4x1x2xf32>) -> tensor<1x4x2x4x1x1xf32>
    %x0_r1 = "ttir.reshape"(%x0_slice) <{shape = [1:i32, 4:i32, 2:i32, 4:i32]}> : (tensor<1x4x2x4x1x1xf32>) -> tensor<1x4x2x4xf32>
    %x0_r2 = "ttir.reshape"(%x0_r1) <{shape = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32]}> : (tensor<1x4x2x4xf32>) -> tensor<1x4x2x4x1xf32>
    %x0_bc = "ttir.broadcast"(%x0_r2) <{broadcast_dimensions = array<i64: 1, 1, 1, 1, 2>}> : (tensor<1x4x2x4x1xf32>) -> tensor<1x4x2x4x2xf32>

    // ---- pair index 1 (imag) ----
    %x1_slice = "ttir.slice_static"(%x_6d) <{begins = [0:i32, 0:i32, 0:i32, 0:i32, 0:i32, 1:i32], ends = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32, 2:i32], step = [1:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x4x2x4x1x2xf32>) -> tensor<1x4x2x4x1x1xf32>
    %x1_r1 = "ttir.reshape"(%x1_slice) <{shape = [1:i32, 4:i32, 2:i32, 4:i32]}> : (tensor<1x4x2x4x1x1xf32>) -> tensor<1x4x2x4xf32>
    %x1_r2 = "ttir.reshape"(%x1_r1) <{shape = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32]}> : (tensor<1x4x2x4xf32>) -> tensor<1x4x2x4x1xf32>
    %x1_bc = "ttir.broadcast"(%x1_r2) <{broadcast_dimensions = array<i64: 1, 1, 1, 1, 2>}> : (tensor<1x4x2x4x1xf32>) -> tensor<1x4x2x4x2xf32>

    // ---- multiply col0 * x_real + col1 * x_imag ----
    %cos_branch = "ttir.multiply"(%f0_bc, %x0_bc) : (tensor<1x4x2x4x2xf32>, tensor<1x4x2x4x2xf32>) -> tensor<1x4x2x4x2xf32>
    %sin_branch = "ttir.multiply"(%f1_bc, %x1_bc) : (tensor<1x4x2x4x2xf32>, tensor<1x4x2x4x2xf32>) -> tensor<1x4x2x4x2xf32>
    %sum = "ttir.add"(%cos_branch, %sin_branch) : (tensor<1x4x2x4x2xf32>, tensor<1x4x2x4x2xf32>) -> tensor<1x4x2x4x2xf32>

    // ---- flatten back to 4D ----
    %result = "ttir.reshape"(%sum) <{shape = [1:i32, 4:i32, 2:i32, 8:i32]}> : (tensor<1x4x2x4x2xf32>) -> tensor<1x4x2x8xf32>
    return %result : tensor<1x4x2x8xf32>
  }
}

// =========================================================================
// Pattern 4: complex-interleaved RoPE (Krea CausalWanModel)
// =========================================================================

// Butterfly (real = a*c - b*d, imag = b*c + a*d) with a,b = even/odd halves of
// x and c,d = cos/sin broadcast over heads, re-interleaved via concat. Anchored
// on the subtract, folds to a single composite.
// CHECK-LABEL: @rope_complex_interleaved_basic
// CHECK: "ttcore.composite"
// CHECK-SAME: composite_name = "rotary_embedding"
module {
  func.func @rope_complex_interleaved_basic(
      %x: tensor<2x2x8xf32>,
      %cos: tensor<2x1x4xf32>,
      %sin: tensor<2x1x4xf32>) -> tensor<2x2x8xf32> {
    // de-interleave x: (2,2,8) -> (2,2,4,2), slice the two pair components
    %x_pairs = "ttir.reshape"(%x) <{shape = [2:i32, 2:i32, 4:i32, 2:i32]}> : (tensor<2x2x8xf32>) -> tensor<2x2x4x2xf32>
    %a_s = "ttir.slice_static"(%x_pairs) <{begins = [0:i32, 0:i32, 0:i32, 0:i32], ends = [2:i32, 2:i32, 4:i32, 1:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<2x2x4x2xf32>) -> tensor<2x2x4x1xf32>
    %a = "ttir.reshape"(%a_s) <{shape = [2:i32, 2:i32, 4:i32]}> : (tensor<2x2x4x1xf32>) -> tensor<2x2x4xf32>
    %b_s = "ttir.slice_static"(%x_pairs) <{begins = [0:i32, 0:i32, 0:i32, 1:i32], ends = [2:i32, 2:i32, 4:i32, 2:i32], step = [1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<2x2x4x2xf32>) -> tensor<2x2x4x1xf32>
    %b = "ttir.reshape"(%b_s) <{shape = [2:i32, 2:i32, 4:i32]}> : (tensor<2x2x4x1xf32>) -> tensor<2x2x4xf32>

    // cos/sin broadcast over the head dim (head-independent freqs table)
    %c = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 2, 1>}> : (tensor<2x1x4xf32>) -> tensor<2x2x4xf32>
    %d = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 2, 1>}> : (tensor<2x1x4xf32>) -> tensor<2x2x4xf32>

    // butterfly: real = a*c - b*d, imag = b*c + a*d
    %ac = "ttir.multiply"(%a, %c) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %bd = "ttir.multiply"(%b, %d) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %real = "ttir.subtract"(%ac, %bd) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %bc = "ttir.multiply"(%b, %c) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %ad = "ttir.multiply"(%a, %d) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %imag = "ttir.add"(%bc, %ad) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>

    // view_as_real: re-interleave real/imag and flatten back to (..., D)
    %real_4d = "ttir.reshape"(%real) <{shape = [2:i32, 2:i32, 4:i32, 1:i32]}> : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %imag_4d = "ttir.reshape"(%imag) <{shape = [2:i32, 2:i32, 4:i32, 1:i32]}> : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %vas = "ttir.concat"(%real_4d, %imag_4d) <{dim = 3 : si32}> : (tensor<2x2x4x1xf32>, tensor<2x2x4x1xf32>) -> tensor<2x2x4x2xf32>
    %result = "ttir.reshape"(%vas) <{shape = [2:i32, 2:i32, 8:i32]}> : (tensor<2x2x4x2xf32>) -> tensor<2x2x8xf32>
    return %result : tensor<2x2x8xf32>
  }
}

// Negative: freqs not broadcast over heads -> x/freqs split is ambiguous -> no fuse.
// CHECK-LABEL: @rope_complex_interleaved_no_broadcast_no_fuse
// CHECK-NOT: "ttcore.composite"
module {
  func.func @rope_complex_interleaved_no_broadcast_no_fuse(
      %a: tensor<2x2x4xf32>, %b: tensor<2x2x4xf32>,
      %c: tensor<2x2x4xf32>, %d: tensor<2x2x4xf32>) -> tensor<2x2x4x2xf32> {
    %ac = "ttir.multiply"(%a, %c) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %bd = "ttir.multiply"(%b, %d) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %real = "ttir.subtract"(%ac, %bd) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %bc = "ttir.multiply"(%b, %c) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %ad = "ttir.multiply"(%a, %d) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %imag = "ttir.add"(%bc, %ad) : (tensor<2x2x4xf32>, tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
    %real_4d = "ttir.reshape"(%real) <{shape = [2:i32, 2:i32, 4:i32, 1:i32]}> : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %imag_4d = "ttir.reshape"(%imag) <{shape = [2:i32, 2:i32, 4:i32, 1:i32]}> : (tensor<2x2x4xf32>) -> tensor<2x2x4x1xf32>
    %vas = "ttir.concat"(%real_4d, %imag_4d) <{dim = 3 : si32}> : (tensor<2x2x4x1xf32>, tensor<2x2x4x1xf32>) -> tensor<2x2x4x2xf32>
    return %vas : tensor<2x2x4x2xf32>
  }
}
