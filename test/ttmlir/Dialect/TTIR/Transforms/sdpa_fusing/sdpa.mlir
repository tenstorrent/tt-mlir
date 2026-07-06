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

// ----------------------------------------------------------------------------
// Kᵀ via ttir.permute (dot_general decomposition form)
//
// Real models lower the score matmul from stablehlo.dot_general, whose
// decomposition expresses Kᵀ as a last-two-dims ttir.permute (not ttir.transpose)
// and applies the K scale *after* the permute. These cases exercise that form.
// ----------------------------------------------------------------------------

// Kᵀ via permute, no scale, no mask.
module {
  func.func @sdpa_kt_via_permute(
      %q: tensor<1x8x128x64xbf16>,
      %k: tensor<1x8x128x64xbf16>,
      %v: tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64xbf16> {
    // CHECK-LABEL: @sdpa_kt_via_permute
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-NOT: ttir.matmul
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x8x128x64xbf16>) -> tensor<1x8x64x128xbf16>
    %s = "ttir.matmul"(%q, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x64xbf16>, tensor<1x8x64x128xbf16>) -> tensor<1x8x128x128xbf16>
    %sm = "ttir.softmax"(%s) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x8x128x128xbf16>) -> tensor<1x8x128x128xbf16>
    %o = "ttir.matmul"(%sm, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x128xbf16>, tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64xbf16>
    return %o : tensor<1x8x128x64xbf16>
  }
}

// Kᵀ via permute with the K scale applied AFTER the permute
// (multiply(permute(K), scale)) — the real-model form.
module {
  func.func @sdpa_kt_via_permute_scale_after(
      %q: tensor<1x8x128x64xf32>,
      %k: tensor<1x8x128x64xf32>,
      %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
    // CHECK-LABEL: @sdpa_kt_via_permute_scale_after
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    // CHECK-NOT: ttir.matmul
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x8x128x64xf32>) -> tensor<1x8x64x128xf32>
    %sc = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 8, 64, 128>}> : () -> tensor<1x8x64x128xf32>
    %kts = "ttir.multiply"(%kt, %sc) : (tensor<1x8x64x128xf32>, tensor<1x8x64x128xf32>) -> tensor<1x8x64x128xf32>
    %s = "ttir.matmul"(%q, %kts) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x64xf32>, tensor<1x8x64x128xf32>) -> tensor<1x8x128x128xf32>
    %sm = "ttir.softmax"(%s) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %o = "ttir.matmul"(%sm, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x128xf32>, tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32>
    return %o : tensor<1x8x128x64xf32>
  }
}

// Full real-model replica: Kᵀ via permute, scale-after-permute, add(mask), and
// the NaN-safe where(rowCond, full(0), softmax) scrub. The scrub must be
// re-applied on the SDPA output.
module {
  func.func @sdpa_real_form_permute_nan_safe(
      %q: tensor<1x8x128x64xf32>,
      %k: tensor<1x8x128x64xf32>,
      %v: tensor<1x8x128x64xf32>,
      %mask: tensor<1x1x128x128xf32>,
      %cond: tensor<1x8x128x1xi1>) -> tensor<1x8x128x64xf32> {
    // CHECK-LABEL: @sdpa_real_form_permute_nan_safe
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    // The NaN-safe select is re-applied on the SDPA output.
    // CHECK: "ttir.where"
    // CHECK-NOT: ttir.matmul
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x8x128x64xf32>) -> tensor<1x8x64x128xf32>
    %sc = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 8, 64, 128>}> : () -> tensor<1x8x64x128xf32>
    %kts = "ttir.multiply"(%kt, %sc) : (tensor<1x8x64x128xf32>, tensor<1x8x64x128xf32>) -> tensor<1x8x64x128xf32>
    %s = "ttir.matmul"(%q, %kts) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x64xf32>, tensor<1x8x64x128xf32>) -> tensor<1x8x128x128xf32>
    %m = "ttir.add"(%s, %mask) : (tensor<1x8x128x128xf32>, tensor<1x1x128x128xf32>) -> tensor<1x8x128x128xf32>
    %sm = "ttir.softmax"(%m) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %z = "ttir.full"() <{fill_value = 0.000000e+00 : f32, shape = array<i32: 1, 8, 128, 128>}> : () -> tensor<1x8x128x128xf32>
    %cb = "ttir.broadcast"(%cond) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x8x128x1xi1>) -> tensor<1x8x128x128xi1>
    %scrub = "ttir.where"(%cb, %z, %sm) : (tensor<1x8x128x128xi1>, tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %o = "ttir.matmul"(%scrub, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x128xf32>, tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32>
    return %o : tensor<1x8x128x64xf32>
  }
}

// Negative: an identity permute is not a last-two-dims transpose, so the score
// is Q·K (not Q·Kᵀ) and must not fuse. Square seq==head_dim keeps the matmul
// valid so only the permute check distinguishes it.
module {
  func.func @sdpa_identity_permute_rejected(
      %q: tensor<1x8x128x128xf32>,
      %k: tensor<1x8x128x128xf32>,
      %v: tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32> {
    // CHECK-LABEL: @sdpa_identity_permute_rejected
    // CHECK-NOT: "ttir.scaled_dot_product_attention"
    // CHECK: ttir.matmul
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 2, 3>}> : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %s = "ttir.matmul"(%q, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %sm = "ttir.softmax"(%s) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    %o = "ttir.matmul"(%sm, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x128x128xf32>, tensor<1x8x128x128xf32>) -> tensor<1x8x128x128xf32>
    return %o : tensor<1x8x128x128xf32>
  }
}

// ----------------------------------------------------------------------------
// Kᵀ folded into the score matmul's transpose_b (PermuteMatmulFusion form)
//
// With enable-permute-matmul-fusion on, PermuteMatmulFusion rewrites
// matmul(Q, permute(K)) into matmul(Q, K, transpose_b=true), absorbing the Kᵀ
// permute. Both forms compute Q·Kᵀ; the folded form has no explicit
// transpose/permute op and its B operand is the un-transposed K, so the matcher
// takes B directly as the key.
// ----------------------------------------------------------------------------

// Folded Kᵀ, no scale, no mask.
module {
  func.func @sdpa_kt_folded_transpose_b(
      %q: tensor<1x32x128x64xbf16>,
      %k: tensor<1x32x128x64xbf16>,
      %v: tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16> {
    // CHECK-LABEL: @sdpa_kt_folded_transpose_b
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.000000e+00
    // CHECK-NOT: ttir.matmul
    %1 = "ttir.matmul"(%q, %k) <{transpose_a = false, transpose_b = true}> : (tensor<1x32x128x64xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x128xbf16>
    %2 = "ttir.softmax"(%1) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %3 = "ttir.matmul"(%2, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    return %3 : tensor<1x32x128x64xbf16>
  }
}

// Full real-model regression replica: folded Kᵀ (transpose_b=true) + post-scale
// via multiply + add(mask). This is exactly what PermuteMatmulFusion produces
// from tt-xla's test_sdpa graph (the uplift regression this path addresses).
module {
  func.func @sdpa_kt_folded_transpose_b_scale_mask(
      %q: tensor<1x8x32x64xbf16>,
      %k: tensor<1x8x32x64xbf16>,
      %v: tensor<1x8x32x64xbf16>,
      %mask: tensor<1x1x32x32xbf16>) -> tensor<1x8x32x64xbf16> {
    // CHECK-LABEL: @sdpa_kt_folded_transpose_b_scale_mask
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    // CHECK-NOT: ttir.matmul
    %1 = "ttir.matmul"(%q, %k) <{transpose_a = false, transpose_b = true}> : (tensor<1x8x32x64xbf16>, tensor<1x8x32x64xbf16>) -> tensor<1x8x32x32xbf16>
    %2 = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %3 = "ttir.multiply"(%1, %2) : (tensor<1x8x32x32xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x8x32x32xbf16>
    %4 = "ttir.add"(%3, %mask) : (tensor<1x8x32x32xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x8x32x32xbf16>
    %5 = "ttir.softmax"(%4) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x8x32x32xbf16>) -> tensor<1x8x32x32xbf16>
    %6 = "ttir.matmul"(%5, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x8x32x32xbf16>, tensor<1x8x32x64xbf16>) -> tensor<1x8x32x64xbf16>
    return %6 : tensor<1x8x32x64xbf16>
  }
}

// ----------------------------------------------------------------------------
// GQA via ttir.repeat_interleave (head-dim expansion)
//
// Models expand K/V from Hkv to Hq heads with a head-dim repeat_interleave
// before the score matmul. SDPA does GQA natively, so the expansion is peeled
// from both K and V and the un-expanded (Hkv-head) tensors feed the op.
// ----------------------------------------------------------------------------

// GQA (Hkv=8, Hq=32, G=4): the native 8-head K/V must reach the op.
module {
  func.func @sdpa_gqa_repeat_interleave(
      %q: tensor<1x32x128x64xf32>,
      %k: tensor<1x8x128x64xf32>,
      %v: tensor<1x8x128x64xf32>) -> tensor<1x32x128x64xf32> {
    // CHECK-LABEL: @sdpa_gqa_repeat_interleave
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>
    // CHECK-NOT: ttir.repeat_interleave
    // CHECK-NOT: ttir.matmul
    %ke = "ttir.repeat_interleave"(%k) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x8x128x64xf32>) -> tensor<1x32x128x64xf32>
    %kt = "ttir.permute"(%ke) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x128x64xf32>) -> tensor<1x32x64x128xf32>
    %s = "ttir.matmul"(%q, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %sm = "ttir.softmax"(%s) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %ve = "ttir.repeat_interleave"(%v) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x8x128x64xf32>) -> tensor<1x32x128x64xf32>
    %o = "ttir.matmul"(%sm, %ve) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x128x64xf32>
    return %o : tensor<1x32x128x64xf32>
  }
}

// Real-model form: repeat_interleave produces bf16, then a typecast upcasts to
// f32 before the score matmul. The head-dim repeat is peeled through the
// typecast, which is re-applied on the native tensor (so K/V stay f32).
module {
  func.func @sdpa_gqa_repeat_interleave_typecast(
      %q: tensor<1x32x128x64xf32>,
      %k: tensor<1x8x128x64xbf16>,
      %v: tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xf32> {
    // CHECK-LABEL: @sdpa_gqa_repeat_interleave_typecast
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>
    // CHECK-NOT: ttir.repeat_interleave
    // CHECK-NOT: ttir.matmul
    %ke = "ttir.repeat_interleave"(%k) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %kef = "ttir.typecast"(%ke) <{conservative_folding = false}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xf32>
    %kt = "ttir.permute"(%kef) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x128x64xf32>) -> tensor<1x32x64x128xf32>
    %s = "ttir.matmul"(%q, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %sm = "ttir.softmax"(%s) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %ve = "ttir.repeat_interleave"(%v) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x8x128x64xbf16>) -> tensor<1x32x128x64xbf16>
    %vef = "ttir.typecast"(%ve) <{conservative_folding = false}> : (tensor<1x32x128x64xbf16>) -> tensor<1x32x128x64xf32>
    %o = "ttir.matmul"(%sm, %vef) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x128x64xf32>
    return %o : tensor<1x32x128x64xf32>
  }
}

// Negative: repeat_interleave on K but not V would make Hkv inconsistent; the
// expansion must NOT be peeled (K stays expanded to 32 to match V).
module {
  func.func @sdpa_gqa_repeat_only_k_not_peeled(
      %q: tensor<1x32x128x64xf32>,
      %k: tensor<1x8x128x64xf32>,
      %v: tensor<1x32x128x64xf32>) -> tensor<1x32x128x64xf32> {
    // CHECK-LABEL: @sdpa_gqa_repeat_only_k_not_peeled
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: tensor<1x32x128x64xf32>, tensor<1x32x128x64xf32>
    %ke = "ttir.repeat_interleave"(%k) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x8x128x64xf32>) -> tensor<1x32x128x64xf32>
    %kt = "ttir.permute"(%ke) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x128x64xf32>) -> tensor<1x32x64x128xf32>
    %s = "ttir.matmul"(%q, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x64xf32>, tensor<1x32x64x128xf32>) -> tensor<1x32x128x128xf32>
    %sm = "ttir.softmax"(%s) <{dimension = -1 : si32, numericStable = false}> : (tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %o = "ttir.matmul"(%sm, %v) <{transpose_a = false, transpose_b = false}> : (tensor<1x32x128x128xf32>, tensor<1x32x128x64xf32>) -> tensor<1x32x128x64xf32>
    return %o : tensor<1x32x128x64xf32>
  }
}

// ----------------------------------------------------------------------------
// Attention sink (softmax padding column)
//
// The sink logit is concat'd as an extra score column before softmax, then the
// column is sliced off after — a NaN-safety pattern that doubles as the sink.
// The sink (broadcast back to [1, Hq, 1, 1]) is fed to the op's attention_sink
// operand. Shape: QKᵀ -> scale -> add(mask) -> concat(sink) -> softmax ->
// slice(drop last col) -> matmul(V).
// ----------------------------------------------------------------------------

// gpt-oss decode form: sink param at KV-head granularity [1,Hkv,1,1] is
// GQA-expanded via repeat_interleave to [1,Hq,1,1] then batch-broadcast.
module {
  func.func @sdpa_attention_sink(
      %q: tensor<2x8x1x64xf32>,
      %k: tensor<2x8x128x64xf32>,
      %v: tensor<2x8x128x64xf32>,
      %mask: tensor<2x1x1x128xf32>,
      %sink: tensor<1x2x1x1xf32>) -> tensor<2x8x1x64xf32> {
    // CHECK-LABEL: @sdpa_attention_sink
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>
    // CHECK-SAME: scale = 1.250000e-01
    // The sink reaches the op as [1, Hq, 1, 1].
    // CHECK-SAME: tensor<1x8x1x1xf32>) -> tensor<2x8x1x64xf32>
    // CHECK-NOT: ttir.concat
    // CHECK-NOT: ttir.slice_static
    // CHECK-NOT: ttir.matmul
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<2x8x128x64xf32>) -> tensor<2x8x64x128xf32>
    %qk = "ttir.matmul"(%q, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<2x8x1x64xf32>, tensor<2x8x64x128xf32>) -> tensor<2x8x1x128xf32>
    %sc = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 2, 8, 1, 128>}> : () -> tensor<2x8x1x128xf32>
    %qks = "ttir.multiply"(%qk, %sc) : (tensor<2x8x1x128xf32>, tensor<2x8x1x128xf32>) -> tensor<2x8x1x128xf32>
    %maskb = "ttir.broadcast"(%mask) <{broadcast_dimensions = array<i64: 1, 8, 1, 1>}> : (tensor<2x1x1x128xf32>) -> tensor<2x8x1x128xf32>
    %qkm = "ttir.add"(%qks, %maskb) : (tensor<2x8x1x128xf32>, tensor<2x8x1x128xf32>) -> tensor<2x8x1x128xf32>
    %sink_exp = "ttir.repeat_interleave"(%sink) <{dim = 1 : si32, repeats = 4 : ui32}> : (tensor<1x2x1x1xf32>) -> tensor<1x8x1x1xf32>
    %sinkb = "ttir.broadcast"(%sink_exp) <{broadcast_dimensions = array<i64: 2, 1, 1, 1>}> : (tensor<1x8x1x1xf32>) -> tensor<2x8x1x1xf32>
    %padded = "ttir.concat"(%qkm, %sinkb) <{dim = 3 : si32}> : (tensor<2x8x1x128xf32>, tensor<2x8x1x1xf32>) -> tensor<2x8x1x129xf32>
    %smx = "ttir.softmax"(%padded) <{dimension = 3 : si32, numericStable = false}> : (tensor<2x8x1x129xf32>) -> tensor<2x8x1x129xf32>
    %trim = "ttir.slice_static"(%smx) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 8 : i32, 1 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x8x1x129xf32>) -> tensor<2x8x1x128xf32>
    %o = "ttir.matmul"(%trim, %v) <{transpose_a = false, transpose_b = false}> : (tensor<2x8x1x128xf32>, tensor<2x8x128x64xf32>) -> tensor<2x8x1x64xf32>
    return %o : tensor<2x8x1x64xf32>
  }
}

// Negative: a slice that does not drop exactly the last column (here it drops
// the FIRST column) is not a sink-padding trim, so it must not fuse.
module {
  func.func @sdpa_sink_wrong_slice_rejected(
      %q: tensor<2x8x1x64xf32>,
      %k: tensor<2x8x128x64xf32>,
      %v: tensor<2x8x128x64xf32>,
      %sink: tensor<1x8x1x1xf32>) -> tensor<2x8x1x64xf32> {
    // CHECK-LABEL: @sdpa_sink_wrong_slice_rejected
    // CHECK-NOT: "ttir.scaled_dot_product_attention"
    // CHECK: ttir.concat
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<2x8x128x64xf32>) -> tensor<2x8x64x128xf32>
    %qk = "ttir.matmul"(%q, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<2x8x1x64xf32>, tensor<2x8x64x128xf32>) -> tensor<2x8x1x128xf32>
    %sinkb = "ttir.broadcast"(%sink) <{broadcast_dimensions = array<i64: 2, 1, 1, 1>}> : (tensor<1x8x1x1xf32>) -> tensor<2x8x1x1xf32>
    %padded = "ttir.concat"(%qk, %sinkb) <{dim = 3 : si32}> : (tensor<2x8x1x128xf32>, tensor<2x8x1x1xf32>) -> tensor<2x8x1x129xf32>
    %smx = "ttir.softmax"(%padded) <{dimension = 3 : si32, numericStable = false}> : (tensor<2x8x1x129xf32>) -> tensor<2x8x1x129xf32>
    %trim = "ttir.slice_static"(%smx) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [2 : i32, 8 : i32, 1 : i32, 129 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x8x1x129xf32>) -> tensor<2x8x1x128xf32>
    %o = "ttir.matmul"(%trim, %v) <{transpose_a = false, transpose_b = false}> : (tensor<2x8x1x128xf32>, tensor<2x8x128x64xf32>) -> tensor<2x8x1x64xf32>
    return %o : tensor<2x8x1x64xf32>
  }
}
