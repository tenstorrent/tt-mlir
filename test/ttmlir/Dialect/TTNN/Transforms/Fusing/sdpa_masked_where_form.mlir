// Pilot 1.3 — SDPA where-form mask fusing (Quetzal §3.4).
//
// Tests three cases for the WhereOp branch in matchScoreComputation:
//  1. Causal where-form (cond is triu(ones, k=0)) -> fuses with is_causal=true
//     and no explicit attention mask.
//  2. Arbitrary boolean cond -> fuses with a materialized additive mask
//     (mask present, is_causal=false).
//  3. Non-(-inf) fill -> does NOT fuse (no ttnn.scaled_dot_product_attention
//     in the output).
//
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=1" %s | FileCheck %s

module {

  // ---------------------------------------------------------------------
  // Case 1: Causal where-form. cond is a structurally-causal mask
  // (upper-triangular with k=0). Expect fusing with is_causal=true.
  // ---------------------------------------------------------------------
  // CHECK-LABEL: func.func @sdpa_where_causal
  // CHECK: "ttnn.scaled_dot_product_attention"
  // CHECK-SAME: is_causal = true
  func.func @sdpa_where_causal(
      %arg0: tensor<1x4x16x16xbf16>,
      %arg1: tensor<1x4x16x16xbf16>,
      %arg2: tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16> {
    // Q @ K^T
    %kT = "ttir.transpose"(%arg1) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %scores = "ttir.matmul"(%arg0, %kT) <{transpose_a = false, transpose_b = false}> : (tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    // Scale.
    %scale = "ttir.full"() <{fill_value = 2.500000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %scaled = "ttir.multiply"(%scores, %scale) : (tensor<1x4x16x16xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x4x16x16xbf16>
    // Causal cond: an upper-triangular boolean mask of all-ones.
    // The dense<...> ElementsAttr below carries the triu pattern.
    %cond = "ttir.constant"() <{value = dense<[
      [[[true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true],
        [false, true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true],
        [false, false, true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true],
        [false, false, false, true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true],
        [false, false, false, false, true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true],
        [false, false, false, false, false, true,  true,  true,  true,  true,  true,  true,  true,  true,  true,  true],
        [false, false, false, false, false, false, true,  true,  true,  true,  true,  true,  true,  true,  true,  true],
        [false, false, false, false, false, false, false, true,  true,  true,  true,  true,  true,  true,  true,  true],
        [false, false, false, false, false, false, false, false, true,  true,  true,  true,  true,  true,  true,  true],
        [false, false, false, false, false, false, false, false, false, true,  true,  true,  true,  true,  true,  true],
        [false, false, false, false, false, false, false, false, false, false, true,  true,  true,  true,  true,  true],
        [false, false, false, false, false, false, false, false, false, false, false, true,  true,  true,  true,  true],
        [false, false, false, false, false, false, false, false, false, false, false, false, true,  true,  true,  true],
        [false, false, false, false, false, false, false, false, false, false, false, false, false, true,  true,  true],
        [false, false, false, false, false, false, false, false, false, false, false, false, false, false, true,  true],
        [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]]]
    ]> : tensor<1x1x16x16xi1>}> : () -> tensor<1x1x16x16xi1>
    %neg_inf = "ttir.full"() <{fill_value = 0xFF800000 : f32, shape = array<i32: 1, 4, 16, 16>}> : () -> tensor<1x4x16x16xbf16>
    // where(cond, -inf, scores): mask out positions where cond is true.
    %masked = "ttir.where"(%cond, %neg_inf, %scaled) : (tensor<1x1x16x16xi1>, tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %sm = "ttir.softmax"(%masked) <{dimension = -1 : si32}> : (tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %out = "ttir.matmul"(%sm, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    return %out : tensor<1x4x16x16xbf16>
  }

  // ---------------------------------------------------------------------
  // Case 2: Arbitrary boolean condition (passed in as %arg3, not a constant
  // triu/tril). Expect fusing with a materialized additive mask;
  // is_causal=false.
  // ---------------------------------------------------------------------
  // CHECK-LABEL: func.func @sdpa_where_arbitrary_cond
  // CHECK: "ttnn.scaled_dot_product_attention"
  // CHECK-SAME: is_causal = false
  func.func @sdpa_where_arbitrary_cond(
      %arg0: tensor<1x4x16x16xbf16>,
      %arg1: tensor<1x4x16x16xbf16>,
      %arg2: tensor<1x4x16x16xbf16>,
      %arg3: tensor<1x1x16x16xi1>) -> tensor<1x4x16x16xbf16> {
    %kT = "ttir.transpose"(%arg1) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %scores = "ttir.matmul"(%arg0, %kT) <{transpose_a = false, transpose_b = false}> : (tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %scale = "ttir.full"() <{fill_value = 2.500000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %scaled = "ttir.multiply"(%scores, %scale) : (tensor<1x4x16x16xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x4x16x16xbf16>
    %neg_inf = "ttir.full"() <{fill_value = 0xFF800000 : f32, shape = array<i32: 1, 4, 16, 16>}> : () -> tensor<1x4x16x16xbf16>
    %masked = "ttir.where"(%arg3, %neg_inf, %scaled) : (tensor<1x1x16x16xi1>, tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %sm = "ttir.softmax"(%masked) <{dimension = -1 : si32}> : (tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %out = "ttir.matmul"(%sm, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    return %out : tensor<1x4x16x16xbf16>
  }

  // ---------------------------------------------------------------------
  // Case 3 (negative): Non-(-inf) fill. The where uses 0.0 instead of -inf
  // for the "masked" branch, which is NOT an attention mask. Expect NO
  // fusion: ttnn.scaled_dot_product_attention must NOT appear.
  // ---------------------------------------------------------------------
  // CHECK-LABEL: func.func @sdpa_where_non_neg_inf_fill
  // CHECK-NOT: ttnn.scaled_dot_product_attention
  func.func @sdpa_where_non_neg_inf_fill(
      %arg0: tensor<1x4x16x16xbf16>,
      %arg1: tensor<1x4x16x16xbf16>,
      %arg2: tensor<1x4x16x16xbf16>,
      %arg3: tensor<1x1x16x16xi1>) -> tensor<1x4x16x16xbf16> {
    %kT = "ttir.transpose"(%arg1) <{dim0 = -2 : si32, dim1 = -1 : si32}> : (tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %scores = "ttir.matmul"(%arg0, %kT) <{transpose_a = false, transpose_b = false}> : (tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %scale = "ttir.full"() <{fill_value = 2.500000e-01 : f32, shape = array<i32: 1, 1, 1, 1>}> : () -> tensor<1x1x1x1xbf16>
    %scaled = "ttir.multiply"(%scores, %scale) : (tensor<1x4x16x16xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x4x16x16xbf16>
    // Fill is 0.0 instead of -inf.
    %fill = "ttir.full"() <{fill_value = 0.000000e+00 : f32, shape = array<i32: 1, 4, 16, 16>}> : () -> tensor<1x4x16x16xbf16>
    %masked = "ttir.where"(%arg3, %fill, %scaled) : (tensor<1x1x16x16xi1>, tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %sm = "ttir.softmax"(%masked) <{dimension = -1 : si32}> : (tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    %out = "ttir.matmul"(%sm, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<1x4x16x16xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    return %out : tensor<1x4x16x16xbf16>
  }
}
