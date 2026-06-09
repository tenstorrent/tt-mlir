// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

// =========================================================================
// Pattern 1: rotate_half RoPE
// =========================================================================

// Basic rotate_half with 4D broadcast cos/sin.
// CHECK-LABEL: @rope_rotate_half_basic
// CHECK: "ttir.rotary_embedding"
// CHECK-NOT: ttir.neg
// CHECK-NOT: ttir.concat
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
// CHECK: "ttir.rotary_embedding"
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
// CHECK: "ttir.rotary_embedding"
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
// CHECK: "ttir.rotary_embedding"
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
// CHECK: "ttir.rotary_embedding"
// CHECK-NOT: ttir.subtract
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

// Rank-3 expanded RoPE as emitted by torch-xla traces: the batch dim is
// dropped, leaving [seq, heads, head_dim]. The op is 4D, so fusion reshapes
// to [batch, heads, seq, head_dim] (seq moved to dim -2) and back.
// Single head (q after tensor-parallel sharding).
// CHECK-LABEL: @rope_expanded_rank3_single_head
// CHECK: "ttir.rotary_embedding"
// CHECK-NOT: ttir.subtract
module {
  func.func @rope_expanded_rank3_single_head(%x: tensor<8x1x64xbf16>, %cos_h: tensor<8x1x32xbf16>, %sin_h: tensor<8x1x32xbf16>) -> tensor<8x1x64xbf16> {
    %first = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32], ends = [8:i32, 1:i32, 32:i32], step = [1:i32, 1:i32, 1:i32]}> : (tensor<8x1x64xbf16>) -> tensor<8x1x32xbf16>
    %second = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 32:i32], ends = [8:i32, 1:i32, 64:i32], step = [1:i32, 1:i32, 1:i32]}> : (tensor<8x1x64xbf16>) -> tensor<8x1x32xbf16>

    %fc = "ttir.multiply"(%first, %cos_h) : (tensor<8x1x32xbf16>, tensor<8x1x32xbf16>) -> tensor<8x1x32xbf16>
    %ss = "ttir.multiply"(%second, %sin_h) : (tensor<8x1x32xbf16>, tensor<8x1x32xbf16>) -> tensor<8x1x32xbf16>
    %sub = "ttir.subtract"(%fc, %ss) : (tensor<8x1x32xbf16>, tensor<8x1x32xbf16>) -> tensor<8x1x32xbf16>

    %sc = "ttir.multiply"(%second, %cos_h) : (tensor<8x1x32xbf16>, tensor<8x1x32xbf16>) -> tensor<8x1x32xbf16>
    %fs = "ttir.multiply"(%first, %sin_h) : (tensor<8x1x32xbf16>, tensor<8x1x32xbf16>) -> tensor<8x1x32xbf16>
    %add = "ttir.add"(%sc, %fs) : (tensor<8x1x32xbf16>, tensor<8x1x32xbf16>) -> tensor<8x1x32xbf16>

    %result = "ttir.concat"(%sub, %add) <{dim = 2 : si32}> : (tensor<8x1x32xbf16>, tensor<8x1x32xbf16>) -> tensor<8x1x64xbf16>
    return %result : tensor<8x1x64xbf16>
  }
}

// Multiple heads (k with grouped KV heads) — exercises the seq/heads swap.
// CHECK-LABEL: @rope_expanded_rank3_multi_head
// CHECK: "ttir.rotary_embedding"
// CHECK-NOT: ttir.subtract
module {
  func.func @rope_expanded_rank3_multi_head(%x: tensor<8x3x64xbf16>, %cos_h: tensor<8x1x32xbf16>, %sin_h: tensor<8x1x32xbf16>) -> tensor<8x3x64xbf16> {
    %cos_bc = "ttir.broadcast"(%cos_h) <{broadcast_dimensions = array<i64: 1, 3, 1>}> : (tensor<8x1x32xbf16>) -> tensor<8x3x32xbf16>
    %sin_bc = "ttir.broadcast"(%sin_h) <{broadcast_dimensions = array<i64: 1, 3, 1>}> : (tensor<8x1x32xbf16>) -> tensor<8x3x32xbf16>

    %first = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 0:i32], ends = [8:i32, 3:i32, 32:i32], step = [1:i32, 1:i32, 1:i32]}> : (tensor<8x3x64xbf16>) -> tensor<8x3x32xbf16>
    %second = "ttir.slice_static"(%x) <{begins = [0:i32, 0:i32, 32:i32], ends = [8:i32, 3:i32, 64:i32], step = [1:i32, 1:i32, 1:i32]}> : (tensor<8x3x64xbf16>) -> tensor<8x3x32xbf16>

    %fc = "ttir.multiply"(%first, %cos_bc) : (tensor<8x3x32xbf16>, tensor<8x3x32xbf16>) -> tensor<8x3x32xbf16>
    %ss = "ttir.multiply"(%second, %sin_bc) : (tensor<8x3x32xbf16>, tensor<8x3x32xbf16>) -> tensor<8x3x32xbf16>
    %sub = "ttir.subtract"(%fc, %ss) : (tensor<8x3x32xbf16>, tensor<8x3x32xbf16>) -> tensor<8x3x32xbf16>

    %sc = "ttir.multiply"(%second, %cos_bc) : (tensor<8x3x32xbf16>, tensor<8x3x32xbf16>) -> tensor<8x3x32xbf16>
    %fs = "ttir.multiply"(%first, %sin_bc) : (tensor<8x3x32xbf16>, tensor<8x3x32xbf16>) -> tensor<8x3x32xbf16>
    %add = "ttir.add"(%sc, %fs) : (tensor<8x3x32xbf16>, tensor<8x3x32xbf16>) -> tensor<8x3x32xbf16>

    %result = "ttir.concat"(%sub, %add) <{dim = 2 : si32}> : (tensor<8x3x32xbf16>, tensor<8x3x32xbf16>) -> tensor<8x3x64xbf16>
    return %result : tensor<8x3x64xbf16>
  }
}

// =========================================================================
// Negative tests
// =========================================================================

// Both halves negated — not valid rotate_half.
// CHECK-LABEL: @rope_neg_both_halves_no_fuse
// CHECK-NOT: "ttir.rotary_embedding"
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
// CHECK-NOT: "ttir.rotary_embedding"
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
// CHECK-NOT: "ttir.rotary_embedding"
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
