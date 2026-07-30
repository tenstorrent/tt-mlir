// Reproduces the CogVideoX-5b DiT attention chain as it actually reaches
// TTIRFusing: the scale arrives as ttir.constant (dense splat) + reshape +
// broadcast, which is what StableHLO-sourced models produce, rather than the
// ttir.full the existing sdpa_fusing tests all use.
//
// Shapes are the per-device (TP=4) CogVideoX ones scaled down so the test is
// cheap; the structure is identical.

// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

// Case A: scale as ttir.constant -> reshape -> broadcast, pre-scale on Q.
// This is the real-model form and the one that currently fails.
module {
  func.func @sdpa_scale_from_constant_prescale_q(
      %q: tensor<2x12x128x64xbf16>,
      %k: tensor<2x12x128x64xbf16>,
      %v: tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16> {
    // CHECK-LABEL: @sdpa_scale_from_constant_prescale_q
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    // CHECK-NOT: ttir.matmul
    %c = "ttir.constant"() <{value = dense<1.250000e-01> : tensor<bf16>}> : () -> tensor<bf16>
    %r = "ttir.reshape"(%c) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
    %b = "ttir.broadcast"(%r) <{broadcast_dimensions = array<i64: 2, 12, 128, 64>}> : (tensor<1x1x1x1xbf16>) -> tensor<2x12x128x64xbf16>
    %qs = "ttir.multiply"(%q, %b) : (tensor<2x12x128x64xbf16>, tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16>
    // Kᵀ as ttir.permute on the last two dims (the dot_general-decomposition form).
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<2x12x128x64xbf16>) -> tensor<2x12x64x128xbf16>
    %s = "ttir.matmul"(%qs, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<2x12x128x64xbf16>, tensor<2x12x64x128xbf16>) -> tensor<2x12x128x128xbf16>
    %p = "ttir.softmax"(%s) <{dimension = -1 : si32, numericStable = true}> : (tensor<2x12x128x128xbf16>) -> tensor<2x12x128x128xbf16>
    %o = "ttir.matmul"(%p, %v) <{transpose_a = false, transpose_b = false}> : (tensor<2x12x128x128xbf16>, tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16>
    return %o : tensor<2x12x128x64xbf16>
  }
}

// Case B: same, but the scale is applied post-matmul on the score tensor.
// This is the form the chain takes after later broadcast/commute passes.
module {
  func.func @sdpa_scale_from_constant_postscale(
      %q: tensor<2x12x128x64xbf16>,
      %k: tensor<2x12x128x64xbf16>,
      %v: tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16> {
    // CHECK-LABEL: @sdpa_scale_from_constant_postscale
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    // CHECK-NOT: ttir.matmul
    %c = "ttir.constant"() <{value = dense<1.250000e-01> : tensor<bf16>}> : () -> tensor<bf16>
    %r = "ttir.reshape"(%c) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
    %b = "ttir.broadcast"(%r) <{broadcast_dimensions = array<i64: 2, 12, 128, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<2x12x128x128xbf16>
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<2x12x128x64xbf16>) -> tensor<2x12x64x128xbf16>
    %s = "ttir.matmul"(%q, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<2x12x128x64xbf16>, tensor<2x12x64x128xbf16>) -> tensor<2x12x128x128xbf16>
    %ss = "ttir.multiply"(%s, %b) : (tensor<2x12x128x128xbf16>, tensor<2x12x128x128xbf16>) -> tensor<2x12x128x128xbf16>
    %p = "ttir.softmax"(%ss) <{dimension = -1 : si32, numericStable = true}> : (tensor<2x12x128x128xbf16>) -> tensor<2x12x128x128xbf16>
    %o = "ttir.matmul"(%p, %v) <{transpose_a = false, transpose_b = false}> : (tensor<2x12x128x128xbf16>, tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16>
    return %o : tensor<2x12x128x64xbf16>
  }
}

// Case C: control -- identical to A but with ttir.full. Must already pass.
module {
  func.func @sdpa_scale_from_full_control(
      %q: tensor<2x12x128x64xbf16>,
      %k: tensor<2x12x128x64xbf16>,
      %v: tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16> {
    // CHECK-LABEL: @sdpa_scale_from_full_control
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-SAME: scale = 1.250000e-01
    // CHECK-NOT: ttir.matmul
    %f = "ttir.full"() <{fill_value = 1.250000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %qs = "ttir.multiply"(%q, %f) : (tensor<2x12x128x64xbf16>, tensor<1x1x1x1xbf16>) -> tensor<2x12x128x64xbf16>
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<2x12x128x64xbf16>) -> tensor<2x12x64x128xbf16>
    %s = "ttir.matmul"(%qs, %kt) <{transpose_a = false, transpose_b = false}> : (tensor<2x12x128x64xbf16>, tensor<2x12x64x128xbf16>) -> tensor<2x12x128x128xbf16>
    %p = "ttir.softmax"(%s) <{dimension = -1 : si32, numericStable = true}> : (tensor<2x12x128x128xbf16>) -> tensor<2x12x128x128xbf16>
    %o = "ttir.matmul"(%p, %v) <{transpose_a = false, transpose_b = false}> : (tensor<2x12x128x128xbf16>, tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16>
    return %o : tensor<2x12x128x64xbf16>
  }
}
