// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// GPT-OSS-20B gate-up SwiGLU subgraph -- small variant for CI (TTIR)
// =============================================================================
//
// Scaled-down version of gate_up_subgraph_0.mlir (4x544x2880 -> 1x32x96)
// for faster CI turnaround.  Same 5-op SwiGLU pattern with broadcast scalars.
//
// Pattern:
//   %0 = ttir.add(value,    value_bias)   // (1x32x96) + (1x1x1)
//   %1 = ttir.multiply(gate, alpha)       // (1x32x96) * (1x1x1)
//   %2 = ttir.sigmoid(%1)
//   %3 = ttir.multiply(gate, %2)
//   %4 = ttir.multiply(%0, %3)
// =============================================================================

module {
  func.func @gpt_oss_20b_gate_up_d2m_subgraph_0(
      %value      : tensor<1x32x96xbf16>,
      %value_bias : tensor<1x1x1xbf16>,
      %gate       : tensor<1x32x96xbf16>,
      %alpha      : tensor<1x1x1xbf16>)
      -> tensor<1x32x96xbf16> {
    %0 = "ttir.add"(%value, %value_bias)
        : (tensor<1x32x96xbf16>, tensor<1x1x1xbf16>) -> tensor<1x32x96xbf16>
    %1 = "ttir.multiply"(%gate, %alpha)
        : (tensor<1x32x96xbf16>, tensor<1x1x1xbf16>) -> tensor<1x32x96xbf16>
    %2 = "ttir.sigmoid"(%1)
        : (tensor<1x32x96xbf16>) -> tensor<1x32x96xbf16>
    %3 = "ttir.multiply"(%gate, %2)
        : (tensor<1x32x96xbf16>, tensor<1x32x96xbf16>) -> tensor<1x32x96xbf16>
    %4 = "ttir.multiply"(%0, %3)
        : (tensor<1x32x96xbf16>, tensor<1x32x96xbf16>) -> tensor<1x32x96xbf16>
    return %4 : tensor<1x32x96xbf16>
  }
}
