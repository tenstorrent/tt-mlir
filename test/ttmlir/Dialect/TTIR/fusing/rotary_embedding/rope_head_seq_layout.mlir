// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

// ttnn.rotary_embedding wants the head axis at index 1 and the sequence axis
// at index 2. Frontends that build Q/K as [B, S, H, D] - diffusers does, via
// unflatten(2, (heads, -1)) - must be swapped into that layout, or OpModel
// vetoes promotion and the composite is inlined back into primitives.
//
// The head axis is the one the cos/sin caches broadcast over, so it is
// recoverable from the cache shape alone.

// =========================================================================
// [B, S, H, D] input: composite is built in [B, H, S, D] and wrapped in a
// matching pair of permutes.
// =========================================================================
// CHECK-LABEL: func.func @rope_bshd_layout
func.func @rope_bshd_layout(%x: tensor<1x64x8x64xf32>, %cos: tensor<1x64x1x64xf32>, %sin: tensor<1x64x1x64xf32>) -> tensor<1x64x8x64xf32> {
  // CHECK: %[[XT:.*]] = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x64x8x64xf32>) -> tensor<1x8x64x64xf32>
  // CHECK: "ttir.permute"(%arg1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x64x1x64xf32>) -> tensor<1x1x64x64xf32>
  // CHECK: "ttir.permute"(%arg2) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x64x1x64xf32>) -> tensor<1x1x64x64xf32>
  // CHECK: %[[C:.*]] = "ttcore.composite"(%[[XT]], {{.*}}) <{composite_name = "rotary_embedding"
  // CHECK-SAME: -> tensor<1x8x64x64xf32>
  // CHECK: %[[OUT:.*]] = "ttir.permute"(%[[C]]) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x8x64x64xf32>) -> tensor<1x64x8x64xf32>
  // CHECK: return %[[OUT]]
  %0 = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 64 : i32, 8 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x8x64xf32>) -> tensor<1x64x8x32xf32>
  %1 = "ttir.neg"(%0) : (tensor<1x64x8x32xf32>) -> tensor<1x64x8x32xf32>
  %2 = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 64 : i32, 8 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x8x64xf32>) -> tensor<1x64x8x32xf32>
  %3 = "ttir.concat"(%1, %2) <{dim = 3 : si32}> : (tensor<1x64x8x32xf32>, tensor<1x64x8x32xf32>) -> tensor<1x64x8x64xf32>
  %4 = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 1, 8, 1>}> : (tensor<1x64x1x64xf32>) -> tensor<1x64x8x64xf32>
  %5 = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 1, 8, 1>}> : (tensor<1x64x1x64xf32>) -> tensor<1x64x8x64xf32>
  %6 = "ttir.multiply"(%x, %4) : (tensor<1x64x8x64xf32>, tensor<1x64x8x64xf32>) -> tensor<1x64x8x64xf32>
  %7 = "ttir.multiply"(%3, %5) : (tensor<1x64x8x64xf32>, tensor<1x64x8x64xf32>) -> tensor<1x64x8x64xf32>
  %8 = "ttir.add"(%6, %7) : (tensor<1x64x8x64xf32>, tensor<1x64x8x64xf32>) -> tensor<1x64x8x64xf32>
  return %8 : tensor<1x64x8x64xf32>
}

// =========================================================================
// [B, H, S, D] input is already canonical: no permutes are introduced.
// =========================================================================
// CHECK-LABEL: func.func @rope_bhsd_layout_unchanged
func.func @rope_bhsd_layout_unchanged(%x: tensor<1x8x64x64xf32>, %cos: tensor<1x1x64x64xf32>, %sin: tensor<1x1x64x64xf32>) -> tensor<1x8x64x64xf32> {
  // CHECK: %[[C:.*]] = "ttcore.composite"(%arg0, %arg1, %arg2) <{composite_name = "rotary_embedding"
  // CHECK-SAME: -> tensor<1x8x64x64xf32>
  // CHECK-NEXT: return %[[C]]
  %0 = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 64 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x64x64xf32>) -> tensor<1x8x64x32xf32>
  %1 = "ttir.neg"(%0) : (tensor<1x8x64x32xf32>) -> tensor<1x8x64x32xf32>
  %2 = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 64 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x64x64xf32>) -> tensor<1x8x64x32xf32>
  %3 = "ttir.concat"(%1, %2) <{dim = 3 : si32}> : (tensor<1x8x64x32xf32>, tensor<1x8x64x32xf32>) -> tensor<1x8x64x64xf32>
  %4 = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x64x64xf32>) -> tensor<1x8x64x64xf32>
  %5 = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<1x1x64x64xf32>) -> tensor<1x8x64x64xf32>
  %6 = "ttir.multiply"(%x, %4) : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>) -> tensor<1x8x64x64xf32>
  %7 = "ttir.multiply"(%3, %5) : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>) -> tensor<1x8x64x64xf32>
  %8 = "ttir.add"(%6, %7) : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>) -> tensor<1x8x64x64xf32>
  return %8 : tensor<1x8x64x64xf32>
}

// =========================================================================
// The interleaved-pair pattern builds its own composite rather than going
// through replaceWithRoPEComposite, so it needs the same swap. Here x is
// [B, S, H, D] = 1x4x2x8 and the derived 4D cache is 1x4x1x8, i.e. full
// extent on the sequence axis and broadcast on the head axis at index 2.
//
// The swap wraps only the composite; the interleave unwind that follows
// (reshape -> permute {0,1,2,4,3} -> reshape) still runs in the original
// layout and is left untouched.
// =========================================================================
// CHECK-LABEL: func.func @rope_interleaved_pair_bshd_layout
func.func @rope_interleaved_pair_bshd_layout(%x: tensor<1x4x2x8xf32>, %freqs: tensor<1x4x1x4x2x2xf32>) -> tensor<1x4x2x8xf32> {
  // x, cos and sin are all permuted into [B, H, S, D] before the composite.
  // CHECK: %[[XT:.*]] = "ttir.permute"({{.*}}) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4x2x8xf32>) -> tensor<1x2x4x8xf32>
  // CHECK: %[[COST:.*]] = "ttir.permute"({{.*}}) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4x1x8xf32>) -> tensor<1x1x4x8xf32>
  // CHECK: %[[SINT:.*]] = "ttir.permute"({{.*}}) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x4x1x8xf32>) -> tensor<1x1x4x8xf32>
  // CHECK: %[[C:.*]] = "ttcore.composite"(%[[XT]], %[[COST]], %[[SINT]]) <{composite_name = "rotary_embedding"
  // CHECK-SAME: (tensor<1x2x4x8xf32>, tensor<1x1x4x8xf32>, tensor<1x1x4x8xf32>) -> tensor<1x2x4x8xf32>
  // CHECK: "ttir.permute"(%[[C]]) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x2x4x8xf32>) -> tensor<1x4x2x8xf32>
  // The interleave unwind still runs in the original layout.
  // CHECK: "ttir.permute"({{.*}}) <{permutation = array<i64: 0, 1, 2, 4, 3>}>

  // col 0 of the freqs_cis 2x2 = [cos, sin]
  %f0_slice = "ttir.slice_static"(%freqs) <{begins = [0:i32, 0:i32, 0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 4:i32, 1:i32, 4:i32, 2:i32, 1:i32], step = [1:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x4x1x4x2x2xf32>) -> tensor<1x4x1x4x2x1xf32>
  %f0_r1 = "ttir.reshape"(%f0_slice) <{shape = [1:i32, 4:i32, 4:i32, 2:i32]}> : (tensor<1x4x1x4x2x1xf32>) -> tensor<1x4x4x2xf32>
  %f0_r2 = "ttir.reshape"(%f0_r1) <{shape = [1:i32, 4:i32, 1:i32, 4:i32, 2:i32]}> : (tensor<1x4x4x2xf32>) -> tensor<1x4x1x4x2xf32>
  %f0_bc = "ttir.broadcast"(%f0_r2) <{broadcast_dimensions = array<i64: 1, 1, 2, 1, 1>}> : (tensor<1x4x1x4x2xf32>) -> tensor<1x4x2x4x2xf32>

  // col 1 of the freqs_cis 2x2 = [-sin, cos]
  %f1_slice = "ttir.slice_static"(%freqs) <{begins = [0:i32, 0:i32, 0:i32, 0:i32, 0:i32, 1:i32], ends = [1:i32, 4:i32, 1:i32, 4:i32, 2:i32, 2:i32], step = [1:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x4x1x4x2x2xf32>) -> tensor<1x4x1x4x2x1xf32>
  %f1_r1 = "ttir.reshape"(%f1_slice) <{shape = [1:i32, 4:i32, 4:i32, 2:i32]}> : (tensor<1x4x1x4x2x1xf32>) -> tensor<1x4x4x2xf32>
  %f1_r2 = "ttir.reshape"(%f1_r1) <{shape = [1:i32, 4:i32, 1:i32, 4:i32, 2:i32]}> : (tensor<1x4x4x2xf32>) -> tensor<1x4x1x4x2xf32>
  %f1_bc = "ttir.broadcast"(%f1_r2) <{broadcast_dimensions = array<i64: 1, 1, 2, 1, 1>}> : (tensor<1x4x1x4x2xf32>) -> tensor<1x4x2x4x2xf32>

  // reshape x to expose pairs: (1,4,2,8) -> (1,4,2,4,1,2)
  %x_6d = "ttir.reshape"(%x) <{shape = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32, 2:i32]}> : (tensor<1x4x2x8xf32>) -> tensor<1x4x2x4x1x2xf32>

  // pair index 0 (real)
  %x0_slice = "ttir.slice_static"(%x_6d) <{begins = [0:i32, 0:i32, 0:i32, 0:i32, 0:i32, 0:i32], ends = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32, 1:i32], step = [1:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x4x2x4x1x2xf32>) -> tensor<1x4x2x4x1x1xf32>
  %x0_r1 = "ttir.reshape"(%x0_slice) <{shape = [1:i32, 4:i32, 2:i32, 4:i32]}> : (tensor<1x4x2x4x1x1xf32>) -> tensor<1x4x2x4xf32>
  %x0_r2 = "ttir.reshape"(%x0_r1) <{shape = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32]}> : (tensor<1x4x2x4xf32>) -> tensor<1x4x2x4x1xf32>
  %x0_bc = "ttir.broadcast"(%x0_r2) <{broadcast_dimensions = array<i64: 1, 1, 1, 1, 2>}> : (tensor<1x4x2x4x1xf32>) -> tensor<1x4x2x4x2xf32>

  // pair index 1 (imag)
  %x1_slice = "ttir.slice_static"(%x_6d) <{begins = [0:i32, 0:i32, 0:i32, 0:i32, 0:i32, 1:i32], ends = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32, 2:i32], step = [1:i32, 1:i32, 1:i32, 1:i32, 1:i32, 1:i32]}> : (tensor<1x4x2x4x1x2xf32>) -> tensor<1x4x2x4x1x1xf32>
  %x1_r1 = "ttir.reshape"(%x1_slice) <{shape = [1:i32, 4:i32, 2:i32, 4:i32]}> : (tensor<1x4x2x4x1x1xf32>) -> tensor<1x4x2x4xf32>
  %x1_r2 = "ttir.reshape"(%x1_r1) <{shape = [1:i32, 4:i32, 2:i32, 4:i32, 1:i32]}> : (tensor<1x4x2x4xf32>) -> tensor<1x4x2x4x1xf32>
  %x1_bc = "ttir.broadcast"(%x1_r2) <{broadcast_dimensions = array<i64: 1, 1, 1, 1, 2>}> : (tensor<1x4x2x4x1xf32>) -> tensor<1x4x2x4x2xf32>

  %cos_branch = "ttir.multiply"(%f0_bc, %x0_bc) : (tensor<1x4x2x4x2xf32>, tensor<1x4x2x4x2xf32>) -> tensor<1x4x2x4x2xf32>
  %sin_branch = "ttir.multiply"(%f1_bc, %x1_bc) : (tensor<1x4x2x4x2xf32>, tensor<1x4x2x4x2xf32>) -> tensor<1x4x2x4x2xf32>
  %sum = "ttir.add"(%cos_branch, %sin_branch) : (tensor<1x4x2x4x2xf32>, tensor<1x4x2x4x2xf32>) -> tensor<1x4x2x4x2xf32>
  %result = "ttir.reshape"(%sum) <{shape = [1:i32, 4:i32, 2:i32, 8:i32]}> : (tensor<1x4x2x4x2xf32>) -> tensor<1x4x2x8xf32>
  return %result : tensor<1x4x2x8xf32>
}

// =========================================================================
// The half-rotation ("stack") form, which is what Wan actually emits once
// tt-xla's _patch_apply_rotary_emb_stack_form is wired in:
//
//   h_p      = h.unflatten(-1,(D/2,2)).transpose(-1,-2).reshape(...)
//   cos      = freqs_cos[..., 0::2] ; sin = freqs_sin[..., 1::2]
//   cos_full = cat([cos, cos], -1)  ; sin_full likewise  (self-concat)
//   rotated  = cat([-second, first], -1)
//   out      = h_p * cos_full + rotated * sin_full
//
// The patch's docstring claims this form does not match the fusing pattern.
// It does: the half-rotation shape IS rotate-half, the input reorganization is
// just part of computing x (both branches of the add share it), and the
// self-concat caches are handled by matchSelfConcatLastDim. The output
// reorganization the docstring also blamed is gone once the patch's C3 change
// (drop the output re-interleave) is applied, which is the state modelled here.
//
// h is [B, S, H, D] = 1x128x8x64, caches [1, S, 1, D] = 1x128x1x64, so this
// takes the head/seq swap too.
// =========================================================================
// CHECK-LABEL: func.func @rope_stack_form_bshd_layout
func.func @rope_stack_form_bshd_layout(%h: tensor<1x128x8x64xbf16>, %fcos: tensor<1x128x1x64xbf16>, %fsin: tensor<1x128x1x64xbf16>) -> tensor<1x128x8x64xbf16> {
  // The deinterleave reorganization is left alone - it feeds the composite.
  // CHECK: "ttir.permute"({{.*}}) <{permutation = array<i64: 0, 1, 2, 4, 3>}>
  // CHECK: %[[HP:.*]] = "ttir.reshape"{{.*}}(tensor<1x128x8x2x32xbf16>) -> tensor<1x128x8x64xbf16>
  // Operands are swapped into [B, H, S, D] and the composite is built there.
  // CHECK: %[[XT:.*]] = "ttir.permute"(%[[HP]]) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x8x64xbf16>) -> tensor<1x8x128x64xbf16>
  // CHECK: %[[COST:.*]] = "ttir.permute"({{.*}}) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x1x64xbf16>) -> tensor<1x1x128x64xbf16>
  // CHECK: %[[SINT:.*]] = "ttir.permute"({{.*}}) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x1x64xbf16>) -> tensor<1x1x128x64xbf16>
  // CHECK: %[[C:.*]] = "ttcore.composite"(%[[XT]], %[[COST]], %[[SINT]]) <{composite_name = "rotary_embedding"
  // CHECK-SAME: (tensor<1x8x128x64xbf16>, tensor<1x1x128x64xbf16>, tensor<1x1x128x64xbf16>) -> tensor<1x8x128x64xbf16>
  // CHECK: %[[OUT:.*]] = "ttir.permute"(%[[C]]) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x8x128x64xbf16>) -> tensor<1x128x8x64xbf16>
  // The main function ends here: every activation-sized multiply, neg and add
  // has been absorbed into the composite. (A CHECK-NOT would scan on into the
  // decomposition function, which legitimately still contains that arithmetic,
  // so CHECK-NEXT on the return is what pins it.)
  // CHECK-NEXT: return %[[OUT]]

  // h_p = deinterleave_last_dim(h)
  %h5 = "ttir.reshape"(%h) <{shape = [1 : i32, 128 : i32, 8 : i32, 32 : i32, 2 : i32]}> : (tensor<1x128x8x64xbf16>) -> tensor<1x128x8x32x2xbf16>
  %hT = "ttir.permute"(%h5) <{permutation = array<i64: 0, 1, 2, 4, 3>}> : (tensor<1x128x8x32x2xbf16>) -> tensor<1x128x8x2x32xbf16>
  %hp = "ttir.reshape"(%hT) <{shape = [1 : i32, 128 : i32, 8 : i32, 64 : i32]}> : (tensor<1x128x8x2x32xbf16>) -> tensor<1x128x8x64xbf16>

  // cos = fcos[..., 0::2] ; sin = fsin[..., 1::2]
  %cos = "ttir.slice_static"(%fcos) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 128 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<1x128x1x64xbf16>) -> tensor<1x128x1x32xbf16>
  %sin = "ttir.slice_static"(%fsin) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [1 : i32, 128 : i32, 1 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}> : (tensor<1x128x1x64xbf16>) -> tensor<1x128x1x32xbf16>

  // self-concat the caches back to full D
  %cosf = "ttir.concat"(%cos, %cos) <{dim = 3 : si32}> : (tensor<1x128x1x32xbf16>, tensor<1x128x1x32xbf16>) -> tensor<1x128x1x64xbf16>
  %sinf = "ttir.concat"(%sin, %sin) <{dim = 3 : si32}> : (tensor<1x128x1x32xbf16>, tensor<1x128x1x32xbf16>) -> tensor<1x128x1x64xbf16>

  // rotated = cat([-second, first], -1)
  %first = "ttir.slice_static"(%hp) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 128 : i32, 8 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x128x8x64xbf16>) -> tensor<1x128x8x32xbf16>
  %second = "ttir.slice_static"(%hp) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 128 : i32, 8 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x128x8x64xbf16>) -> tensor<1x128x8x32xbf16>
  %negsec = "ttir.neg"(%second) : (tensor<1x128x8x32xbf16>) -> tensor<1x128x8x32xbf16>
  %rot = "ttir.concat"(%negsec, %first) <{dim = 3 : si32}> : (tensor<1x128x8x32xbf16>, tensor<1x128x8x32xbf16>) -> tensor<1x128x8x64xbf16>

  // out = h_p * cos_full + rotated * sin_full
  %m1 = "ttir.multiply"(%hp, %cosf) : (tensor<1x128x8x64xbf16>, tensor<1x128x1x64xbf16>) -> tensor<1x128x8x64xbf16>
  %m2 = "ttir.multiply"(%rot, %sinf) : (tensor<1x128x8x64xbf16>, tensor<1x128x1x64xbf16>) -> tensor<1x128x8x64xbf16>
  %out = "ttir.add"(%m1, %m2) : (tensor<1x128x8x64xbf16>, tensor<1x128x8x64xbf16>) -> tensor<1x128x8x64xbf16>
  return %out : tensor<1x128x8x64xbf16>
}

// =========================================================================
// Fully broadcast caches are ambiguous - both candidate axes are size 1 -
// so the layout is read as canonical and left alone.
// =========================================================================
// CHECK-LABEL: func.func @rope_ambiguous_cache_unchanged
func.func @rope_ambiguous_cache_unchanged(%x: tensor<1x8x8x64xf32>, %cos: tensor<1x1x1x64xf32>, %sin: tensor<1x1x1x64xf32>) -> tensor<1x8x8x64xf32> {
  // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2) <{composite_name = "rotary_embedding"
  // CHECK-NOT: "ttir.permute"
  %0 = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 32 : i32], ends = [1 : i32, 8 : i32, 8 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x32xf32>
  %1 = "ttir.neg"(%0) : (tensor<1x8x8x32xf32>) -> tensor<1x8x8x32xf32>
  %2 = "ttir.slice_static"(%x) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 8 : i32, 32 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x8x64xf32>) -> tensor<1x8x8x32xf32>
  %3 = "ttir.concat"(%1, %2) <{dim = 3 : si32}> : (tensor<1x8x8x32xf32>, tensor<1x8x8x32xf32>) -> tensor<1x8x8x64xf32>
  %4 = "ttir.broadcast"(%cos) <{broadcast_dimensions = array<i64: 1, 8, 8, 1>}> : (tensor<1x1x1x64xf32>) -> tensor<1x8x8x64xf32>
  %5 = "ttir.broadcast"(%sin) <{broadcast_dimensions = array<i64: 1, 8, 8, 1>}> : (tensor<1x1x1x64xf32>) -> tensor<1x8x8x64xf32>
  %6 = "ttir.multiply"(%x, %4) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %7 = "ttir.multiply"(%3, %5) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  %8 = "ttir.add"(%6, %7) : (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
  return %8 : tensor<1x8x8x64xf32>
}
