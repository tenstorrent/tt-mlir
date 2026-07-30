// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Faithful reproduction of the CogVideoX-5b DiT attention chain as it actually
// enters the TTIR pipeline from StableHLO. Unlike sdpa.mlir, this does NOT
// start from a pre-fused ttir.softmax / ttir.matmul -- it starts from
// ttir.dot_general plus the raw numerically-stable softmax decomposition, which
// is what a framework-traced model presents, and runs the same pass order the
// real pipeline uses (fusing -> decomposition -> canonicalize -> fusing).
//
// Op-for-op transcription of the real graph (shapes reduced from
// 2x12x17776x{64,17776} so the test is cheap; structure identical):
//   scale : ttir.constant(splat) -> reshape -> broadcast -> multiply on Q
//   Kᵀ    : ttir.permute [0, 1, 3, 2]
//   scores: ttir.dot_general batch=[0,1] contract=lhs[3]/rhs[2]
//   softmax: max(keep_dim=false) -> reshape -> broadcast -> subtract -> exp
//            -> sum(keep_dim=false) -> reshape -> broadcast -> div
//   out   : ttir.dot_general batch=[0,1] contract=lhs[3]/rhs[2]

// RUN: ttmlir-opt --ttir-fusing --ttir-to-ttir-decomposition --canonicalize --ttir-fusing %s | FileCheck %s

module {
  func.func @sdpa_from_dot_general_real_form(
      %q: tensor<2x12x128x64xbf16>,
      %k: tensor<2x12x128x64xbf16>,
      %v: tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16> {
    // CHECK-LABEL: @sdpa_from_dot_general_real_form
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-NOT: ttir.matmul
    // CHECK-NOT: ttir.softmax

    // Q * (64 ** -0.5), scale as a splat ttir.constant broadcast to full shape.
    %c = "ttir.constant"() <{value = dense<1.250000e-01> : tensor<bf16>}> : () -> tensor<bf16>
    %cr = "ttir.reshape"(%c) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
    %cb = "ttir.broadcast"(%cr) <{broadcast_dimensions = array<i64: 2, 12, 128, 64>}> : (tensor<1x1x1x1xbf16>) -> tensor<2x12x128x64xbf16>
    %qs = "ttir.multiply"(%q, %cb) : (tensor<2x12x128x64xbf16>, tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16>

    // Kᵀ
    %kt = "ttir.permute"(%k) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<2x12x128x64xbf16>) -> tensor<2x12x64x128xbf16>

    // scores = Q·Kᵀ
    %scores = "ttir.dot_general"(%qs, %kt) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<2x12x128x64xbf16>, tensor<2x12x64x128xbf16>) -> tensor<2x12x128x128xbf16>

    // numerically-stable softmax, fully decomposed
    %mx = "ttir.max"(%scores) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x12x128x128xbf16>) -> tensor<2x12x128xbf16>
    %mxr = "ttir.reshape"(%mx) <{shape = [2 : i32, 12 : i32, 128 : i32, 1 : i32]}> : (tensor<2x12x128xbf16>) -> tensor<2x12x128x1xbf16>
    %mxb = "ttir.broadcast"(%mxr) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<2x12x128x1xbf16>) -> tensor<2x12x128x128xbf16>
    %sub = "ttir.subtract"(%scores, %mxb) : (tensor<2x12x128x128xbf16>, tensor<2x12x128x128xbf16>) -> tensor<2x12x128x128xbf16>
    %ex = "ttir.exp"(%sub) : (tensor<2x12x128x128xbf16>) -> tensor<2x12x128x128xbf16>
    %sm = "ttir.sum"(%ex) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x12x128x128xbf16>) -> tensor<2x12x128xbf16>
    %smr = "ttir.reshape"(%sm) <{shape = [2 : i32, 12 : i32, 128 : i32, 1 : i32]}> : (tensor<2x12x128xbf16>) -> tensor<2x12x128x1xbf16>
    %smb = "ttir.broadcast"(%smr) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<2x12x128x1xbf16>) -> tensor<2x12x128x128xbf16>
    %probs = "ttir.div"(%ex, %smb) : (tensor<2x12x128x128xbf16>, tensor<2x12x128x128xbf16>) -> tensor<2x12x128x128xbf16>

    // out = probs·V
    %out = "ttir.dot_general"(%probs, %v) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<2x12x128x128xbf16>, tensor<2x12x128x64xbf16>) -> tensor<2x12x128x64xbf16>
    return %out : tensor<2x12x128x64xbf16>
  }
}
