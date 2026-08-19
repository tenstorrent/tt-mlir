// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// End-to-end: the DiT adaLN gated-residual epilogue expressed as primitive
// TTIR ops (matmul + multiply + add) is fused at the TTNN level (ttnn-fusing)
// into a single ttnn.dit_matmul_addcmul_fused. No TTIR op is involved.

// Matmul variant (no bias).
// CHECK-LABEL: func.func @dit_matmul_addcmul_no_bias
// CHECK: "ttnn.dit_matmul_addcmul_fused"
// CHECK-NOT: "ttnn.matmul"
// CHECK-NOT: "ttnn.multiply"
module {
  func.func @dit_matmul_addcmul_no_bias(%a: tensor<32x128xbf16>, %b: tensor<128x256xbf16>,
                                        %gate: tensor<32x256xbf16>, %res: tensor<32x256xbf16>)
      -> tensor<32x256xbf16> {
    %0 = "ttir.matmul"(%a, %b) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x128xbf16>, tensor<128x256xbf16>) -> tensor<32x256xbf16>
    %1 = "ttir.multiply"(%0, %gate) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
    %2 = "ttir.add"(%1, %res) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
    return %2 : tensor<32x256xbf16>
  }
}

// -----

// Linear variant (with bias): the bias folds into the projection and rides into
// the fused op as its optional operand.
// CHECK-LABEL: func.func @dit_matmul_addcmul_with_bias
// CHECK: "ttnn.dit_matmul_addcmul_fused"
module {
  func.func @dit_matmul_addcmul_with_bias(%a: tensor<32x128xbf16>, %b: tensor<128x256xbf16>,
                                          %bias: tensor<1x256xbf16>,
                                          %gate: tensor<32x256xbf16>, %res: tensor<32x256xbf16>)
      -> tensor<32x256xbf16> {
    %0 = "ttir.linear"(%a, %b, %bias) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x128xbf16>, tensor<128x256xbf16>, tensor<1x256xbf16>) -> tensor<32x256xbf16>
    %1 = "ttir.multiply"(%gate, %0) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
    %2 = "ttir.add"(%res, %1) : (tensor<32x256xbf16>, tensor<32x256xbf16>) -> tensor<32x256xbf16>
    return %2 : tensor<32x256xbf16>
  }
}

// -----

// WAN 2.2 DiT shape: `ttir.dot_general` lowers to a 2D matmul plus a
// unit-outer-dim reshape back to the rank-3 shape the adaLN epilogue runs at.
// The reshape is folded out of the way and replayed on the fused result, so the
// whole epilogue still collapses into one op.
// CHECK-LABEL: func.func @dit_dot_general_addcmul_rank3
// CHECK: %[[FUSED:.*]] = "ttnn.dit_matmul_addcmul_fused"
// CHECK: "ttnn.reshape"(%[[FUSED]])
// CHECK-SAME: -> tensor<1x32x256xbf16
// CHECK-NOT: "ttnn.matmul"
// CHECK-NOT: "ttnn.linear"
// CHECK-NOT: "ttnn.multiply"
module {
  func.func @dit_dot_general_addcmul_rank3(%a: tensor<1x32x128xbf16>, %b: tensor<128x256xbf16>,
                                           %gate: tensor<1x1x256xbf16>, %res: tensor<1x32x256xbf16>)
      -> tensor<1x32x256xbf16> {
    %0 = "ttir.dot_general"(%a, %b) <{batch_dims_lhs = array<i64>, contract_dims_lhs = array<i64: 2>,
                                      batch_dims_rhs = array<i64>, contract_dims_rhs = array<i64: 0>}>
        : (tensor<1x32x128xbf16>, tensor<128x256xbf16>) -> tensor<1x32x256xbf16>
    %1 = "ttir.multiply"(%0, %gate) : (tensor<1x32x256xbf16>, tensor<1x1x256xbf16>) -> tensor<1x32x256xbf16>
    %2 = "ttir.add"(%res, %1) : (tensor<1x32x256xbf16>, tensor<1x32x256xbf16>) -> tensor<1x32x256xbf16>
    return %2 : tensor<1x32x256xbf16>
  }
}
