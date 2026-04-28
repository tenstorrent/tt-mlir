// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// GPT-OSS-20B gate-up SwiGLU subgraph -- d2m PCC repro (TTIR)
// =============================================================================
//
// This is a TTIR encoding of `@d2m_subgraph_0` from the GPT-OSS-20B model dump
// (graph @SyncTensorsGraph.12689, "IR Dump After TTNND2MFusing"). It is the
// SwiGLU activation + value bias residual that the GPT-OSS MLP fuses 24x (one
// per transformer layer); when D2MElementwiseFusion collapses all 5 ops into a
// single d2m.generic, PCC drops from ~0.996 to ~0.480 on test_gpt_oss_mlp.
//
// Pattern (5 elementwise ops, equivalent to (value+vbias) * gate * sigmoid(alpha*gate)):
//   %0 = ttir.add(value,    value_bias)   // (4x544x2880) + (1x1x1)  -- implicit broadcast
//   %1 = ttir.multiply(gate, alpha)       // (4x544x2880) * (1x1x1)  [alpha = 1.702]
//   %2 = ttir.sigmoid(%1)
//   %3 = ttir.multiply(gate, %2)
//   %4 = ttir.multiply(%0, %3)            // == d2m_subgraph_0 result
//
// TTIR's elementwise ops carry the `TTIR_Broadcastable` trait, so the
// (4x544x2880) + (1x1x1) signature is legal at the TTIR level (verifier infers
// the result via NumPy-style broadcast rules) and lowers to ttnn.add / ttnn.multiply
// with implicit-broadcast operands -- matching the dump structure 1:1, no
// ttir.broadcast wrapper needed.
//
// Inputs are sized to match a single TP shard of GPT-OSS-20B prefill
// (batch=4, seq_len=544, hidden=2880) so the produced d2m.generic has the
// same grid / blocking / DST footprint as in the full model.
//
// Source TTNN-form snippet (with #ttnn_layout aliases):
//   /localdev/brapanan/d2m-pcc-repro/extracted/
//       prefill_after_ttnn_d2m_fusing_subgraph0_TTNN.mlir
// =============================================================================

module {
  func.func @gpt_oss_20b_gate_up_d2m_subgraph_0(
      %value      : tensor<4x544x2880xbf16>,
      %value_bias : tensor<1x1x1xbf16>,
      %gate       : tensor<4x544x2880xbf16>,
      %alpha      : tensor<1x1x1xbf16>)
      -> tensor<4x544x2880xbf16> {
    %0 = "ttir.add"(%value, %value_bias)
        : (tensor<4x544x2880xbf16>, tensor<1x1x1xbf16>) -> tensor<4x544x2880xbf16>
    %1 = "ttir.multiply"(%gate, %alpha)
        : (tensor<4x544x2880xbf16>, tensor<1x1x1xbf16>) -> tensor<4x544x2880xbf16>
    %2 = "ttir.sigmoid"(%1)
        : (tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    %3 = "ttir.multiply"(%gate, %2)
        : (tensor<4x544x2880xbf16>, tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    %4 = "ttir.multiply"(%0, %3)
        : (tensor<4x544x2880xbf16>, tensor<4x544x2880xbf16>) -> tensor<4x544x2880xbf16>
    return %4 : tensor<4x544x2880xbf16>
  }
}
