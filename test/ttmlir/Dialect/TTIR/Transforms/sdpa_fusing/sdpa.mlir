// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

// Test the canonical mathematical form of SDPA fusing at the TTIR level.
// Inputs are pre-softmax-fused (use ttir.softmax directly) unless a case
// is explicitly testing the integration with SoftmaxFusionPattern.

// ----------------------------------------------------------------------------
// Basic shapes / scale forms
// ----------------------------------------------------------------------------

// MHA, no scale, no mask.
module {
  func.func @sdpa_mha_no_scale_no_mask(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_no_scale_no_mask
    // CHECK: "ttir.scaled_dot_product_attention"
    // An unscaled source must emit an explicit scale = 1.0, NOT an omitted
    // scale (which the op would interpret as the default 1/sqrt(D)).
    // CHECK-SAME: scale = 1.000000e+00
    // CHECK-NOT: ttir.matmul
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.matmul"(%2, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %3 : tensor<1x32x128x64xbf16>
  }
}

// MHA + add(mask).
module {
  func.func @sdpa_mha_with_mask(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>,
      %mask: tensor<1x1x128x128xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_with_mask
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.000000e+00
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.add"(%1, %mask) : (tensor<1x32x128x128xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.softmax"(%2) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = "ttir.matmul"(%3, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %4 : tensor<1x32x128x64xbf16>
  }
}

// Post-matmul scale via multiply.
module {
  func.func @sdpa_mha_post_scale_mul(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>,
      %mask: tensor<1x1x128x128xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_post_scale_mul
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %3 = "ttir.multiply"(%1, %2) : (tensor<1x32x128x128xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = "ttir.add"(%3, %mask) : (tensor<1x32x128x128xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %5 = "ttir.softmax"(%4) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %6 = "ttir.matmul"(%5, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %6 : tensor<1x32x128x64xbf16>
  }
}

// Post-matmul scale via divide (1/8 = 0.125).
module {
  func.func @sdpa_mha_post_scale_div(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_post_scale_div
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.full"() <{fill_value = 8.000000e+00 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %3 = "ttir.div"(%1, %2) : (tensor<1x32x128x128xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = "ttir.softmax"(%3) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %5 = "ttir.matmul"(%4, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %5 : tensor<1x32x128x64xbf16>
  }
}

// Pre-scale on Q via multiply.
module {
  func.func @sdpa_mha_pre_scale_q_mul(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_pre_scale_q_mul
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    %s = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %qs = "ttir.multiply"(%q, %s) : (tensor<1x32x128x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x64xbf16>
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%qs, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.matmul"(%2, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %3 : tensor<1x32x128x64xbf16>
  }
}

// Pre-scale on Q via divide (1/8 = 0.125).
module {
  func.func @sdpa_mha_pre_scale_q_div(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_pre_scale_q_div
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    %s = "ttir.full"() <{fill_value = 8.000000e+00 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %qs = "ttir.div"(%q, %s) : (tensor<1x32x128x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x64xbf16>
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%qs, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.matmul"(%2, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %3 : tensor<1x32x128x64xbf16>
  }
}

// Pre-scale on K (before the transpose).
module {
  func.func @sdpa_mha_pre_scale_k_mul(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_pre_scale_k_mul
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    %s = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %ks = "ttir.multiply"(%k, %s) : (tensor<1x32x128x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x64xbf16>
    %0 = "ttir.transpose"(%ks) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.matmul"(%2, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %3 : tensor<1x32x128x64xbf16>
  }
}

// Double-scale (pre-Q AND post-matmul) is rejected as ambiguous.
module {
  func.func @sdpa_mha_double_scale_rejected(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_double_scale_rejected
    // CHECK-NOT: ttir.scaled_dot_product_attention
    %sq = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %qs = "ttir.multiply"(%q, %sq) : (tensor<1x32x128x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x64xbf16>
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%qs, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %sp = "ttir.full"() <{fill_value = 2.500000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %2 = "ttir.multiply"(%1, %sp) : (tensor<1x32x128x128xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.softmax"(%2) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = "ttir.matmul"(%3, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %4 : tensor<1x32x128x64xbf16>
  }
}

// ----------------------------------------------------------------------------
// Structural typecasts at the softmax precision boundary
// ----------------------------------------------------------------------------

// Softmax computed in f32; scores and result in bf16. Two typecasts wrap the
// softmax: one in (bf16 -> f32) and one out (f32 -> bf16). The matcher must
// peel both at their named positions.
module {
  func.func @sdpa_mha_softmax_f32_promotion(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>,
      %mask: tensor<1x1x128x128xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_softmax_f32_promotion
    // CHECK: "ttir.scaled_dot_product_attention"
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %s = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %2 = "ttir.multiply"(%1, %s) : (tensor<1x32x128x128xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.add"(%2, %mask) : (tensor<1x32x128x128xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = "ttir.typecast"(%3) <{conservative_folding = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xf32>
    %5 = "ttir.softmax"(%4) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %6 = "ttir.typecast"(%5) <{conservative_folding = false}> : (tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xbf16>
    %7 = "ttir.matmul"(%6, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %7 : tensor<1x32x128x64xbf16>
  }
}

// Typecast on the scale constant is absorbed by extractConstant.
module {
  func.func @sdpa_mha_typecast_on_scale_const(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_typecast_on_scale_const
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    %sf32 = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xf32>
    %s = "ttir.typecast"(%sf32) <{conservative_folding = false}> : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xbf16>
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.multiply"(%1, %s) : (tensor<1x32x128x128xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.softmax"(%2) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = "ttir.matmul"(%3, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %4 : tensor<1x32x128x64xbf16>
  }
}

// ----------------------------------------------------------------------------
// MHA / MQA / GQA / decode shape
// ----------------------------------------------------------------------------

// MQA: Hkv == 1, Hq == 32.
module {
  func.func @sdpa_mqa(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x1x128x64xbf16>,
      %v: tensor<1x1x128x64xbf16>,
      %mask: tensor<1x1x128x128xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mqa
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.000000e+00
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x1x128x64xbf16>) -> tensor<1x1x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x1x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.add"(%1, %mask) : (tensor<1x32x128x128xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.softmax"(%2) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = "ttir.matmul"(%3, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x1x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %4 : tensor<1x32x128x64xbf16>
  }
}

// GQA via repeat_interleave is intentionally NOT supported in v1 — see plan.
// To exercise it, the frontend must expand K/V to Hq with reshape/broadcast or
// repeat_interleave; the v1 matcher does not look through those, so GQA chains
// will fall through to the existing TTNN-level matcher as a fallback.

// Sq==1: the decode shape produces a single ttir SDPA op; decode-specific
// permutes appear only later, in TTIRToTTNN conversion.
module {
  func.func @sdpa_decode_shape_sq1(
      %q: tensor<1x32x1x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>,
      %mask: tensor<1x1x1x128xbf16>) -> tensor<1x32x1x64xbf16> {
    // CHECK-LABEL: @sdpa_decode_shape_sq1
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.000000e+00
    // CHECK-NOT: ttir.permute
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x1x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x1x128xbf16>
    %2 = "ttir.add"(%1, %mask) : (tensor<1x32x1x128xbf16>, tensor<1x1x1x128xbf16>) -> tensor<1x32x1x128xbf16>
    %3 = "ttir.softmax"(%2) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x1x128xbf16>) -> tensor<1x32x1x128xbf16>
    %4 = "ttir.matmul"(%3, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x1x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x1x64xbf16>
    return %4 : tensor<1x32x1x64xbf16>
  }
}

// ----------------------------------------------------------------------------
// Negative cases — the matcher should decline.
// ----------------------------------------------------------------------------

// K is fed via matmul transpose_b instead of an explicit ttir.transpose.
// v1 of the TTIR matcher intentionally does NOT match this form.
module {
  func.func @sdpa_k_transpose_b_rejected(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_k_transpose_b_rejected
    // CHECK-NOT: ttir.scaled_dot_product_attention
    %1 = "ttir.matmul"(%q, %k) <{transpose_a = false, transpose_b = true}> : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.matmul"(%2, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %3 : tensor<1x32x128x64xbf16>
  }
}

// 3D Q/K/V are rejected (v1 requires strict 4D, no implicit unsqueezing).
module {
  func.func @sdpa_3d_rejected(
      %q: tensor<1x128x64xbf16>,
      %k: tensor<1x128x64xbf16>,
      %v: tensor<1x128x64xbf16>) -> tensor<1x128x64xbf16> {
    // CHECK-LABEL: @sdpa_3d_rejected
    // CHECK-NOT: ttir.scaled_dot_product_attention
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x128x64xbf16>) -> tensor<1x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x128x64xbf16>, tensor<1x64x128xbf16>) -> tensor<1x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x128x128xbf16>) -> tensor<1x128x128xbf16>
    %3 = "ttir.matmul"(%2, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x128x128xbf16>, tensor<1x128x64xbf16>) -> tensor<1x128x64xbf16>
    return %3 : tensor<1x128x64xbf16>
  }
}

// Softmax along the wrong dim (not the last/kv axis) is rejected.
module {
  func.func @sdpa_wrong_softmax_dim_rejected(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_wrong_softmax_dim_rejected
    // CHECK-NOT: ttir.scaled_dot_product_attention
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xbf16>, tensor<1x32x64x128xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = 1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.matmul"(%2, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %3 : tensor<1x32x128x64xbf16>
  }
}

// ----------------------------------------------------------------------------
// NaN-safe softmax (where(row_all_masked, 0, softmax))
// ----------------------------------------------------------------------------

// A fully-masked query row makes softmax produce NaN; models wrap the softmax
// in where(row_all_masked, 0, softmax) to scrub it. SDPA itself is NOT NaN-safe
// (matches PyTorch / tt-metal), so the matcher fuses the core and re-applies the
// row-zeroing where to the SDPA output: zeroing whole rows of the attention
// weights == zeroing the same output rows (the matmul contracts over the kv
// axis). The row predicate is broadcast across the kv axis, so it is sound to
// move it past the matmul onto the output head dim.
module {
  func.func @sdpa_mha_nan_safe_where(
      %q: tensor<1x8x128x64xbf16>,
      %k: tensor<1x8x128x64xbf16>,
      %v: tensor<1x8x128x64xbf16>,
      %rowcond: tensor<1x8x128x1xi1>) -> tensor<1x8x128x64xbf16> {
    // CHECK-LABEL: @sdpa_mha_nan_safe_where
    // CHECK: %[[SDPA:[0-9a-z_]+]] = "ttir.scaled_dot_product_attention"
    // CHECK: "ttir.where"(%{{.*}}, %{{.*}}, %[[SDPA]])
    // CHECK-NOT: ttir.matmul
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x64xbf16>, tensor<1x8x64x128xbf16>) -> tensor<1x8x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x8x128x128xbf16>) -> tensor<1x8x128x128xbf16>
    %cond = "ttir.broadcast"(%rowcond) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x8x128x1xi1>) -> tensor<1x8x128x128xi1>
    %zeros = "ttir.zeros"() <{shape = array<i32: 1, 8, 128, 128>}> : () -> tensor<1x8x128x128xbf16>
    %safe = "ttir.where"(%cond, %zeros, %2) : (tensor<1x8x128x128xi1>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>) -> tensor<1x8x128x128xbf16>
    %3 = "ttir.matmul"(%safe, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x128xbf16>, tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64xbf16>
    return %3 : tensor<1x8x128x64xbf16>
  }
}

// Soundness guard: a where whose condition varies across the kv (last) axis
// does NOT zero whole query rows, so it cannot be moved past the matmul. The
// matcher must decline (leave it for the TTNN fallback) rather than fuse.
module {
  func.func @sdpa_where_not_row_uniform_rejected(
      %q: tensor<1x8x128x64xbf16>,
      %k: tensor<1x8x128x64xbf16>,
      %v: tensor<1x8x128x64xbf16>,
      %cond: tensor<1x8x128x128xi1>) -> tensor<1x8x128x64xbf16> {
    // CHECK-LABEL: @sdpa_where_not_row_uniform_rejected
    // CHECK-NOT: ttir.scaled_dot_product_attention
    %0 = "ttir.transpose"(%k) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x64x128xbf16>
    %1 = "ttir.matmul"(%q, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x64xbf16>, tensor<1x8x64x128xbf16>) -> tensor<1x8x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x8x128x128xbf16>) -> tensor<1x8x128x128xbf16>
    %zeros = "ttir.zeros"() <{shape = array<i32: 1, 8, 128, 128>}> : () -> tensor<1x8x128x128xbf16>
    %safe = "ttir.where"(%cond, %zeros, %2) : (tensor<1x8x128x128xi1>, tensor<1x8x128x128xbf16>, tensor<1x8x128x128xbf16>) -> tensor<1x8x128x128xbf16>
    %3 = "ttir.matmul"(%safe, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x128xbf16>, tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64xbf16>
    return %3 : tensor<1x8x128x64xbf16>
  }
}
