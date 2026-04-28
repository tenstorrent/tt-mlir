// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// GPT-OSS-20B gate-up subgraph
// =============================================================================
// Seeing actual_pcc=0.5199745069515318 when running test test_fusion_with_optimizer.py
// on this snippet.

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
