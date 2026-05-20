// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// Regression test: when an input-operand workaround flips a tensor from DRAM
// Interleaved RowMajor to L1 Interleaved RowMajor, the inserted ttnn.to_layout
// op's result encoding must use the canonical L1-Interleaved memref form
// (calculateLogicalShardShapeForL1Interleaved -> <1 x tensorVolume / numCores>),
// not the source layout's shard shape.
//
// Without canonicalization the encoding ends up as memref<32x128xbf16, l1>
// (preserved from source); downstream passes (TTNNDecomposeLayouts)
// then rebuild the type via the canonical path and produce <1x4096xbf16, l1>,
// causing function-return verifier failures on const-eval functions.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// DRAM Interleaved layouts (canonical: memref shape == tensor shape on 1x1 grid).
#dram_input  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x128xbf16, #dram>, <interleaved>>
#dram_idx    = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x4xui16, #dram>, <interleaved>>
#dram_score  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x4xbf16, #dram>, <interleaved>>
#dram_map    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x4xui16, #dram>, <interleaved>>

#dram_disp_out = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x128xbf16, #dram>, <interleaved>>
#dram_idx_out  = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x4xui16, #dram>, <interleaved>>
#dram_sc_out   = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x4xbf16, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: func.func @dispatch_metadata_l1_canonical_form

  // input_tensor: 32 x 128 = 4096 -> memref<1x4096xbf16, l1>
  // CHECK: "ttnn.to_layout"
  // CHECK-SAME: -> tensor<{{.*}}xbf16, #ttnn.ttnn_layout<{{.*}} memref<1x4096xbf16, #ttnn.buffer_type<l1>>, <interleaved>>>

  // expert_indices: 32 x 4 = 128 -> memref<1x128xui16, l1>
  // CHECK: "ttnn.to_layout"
  // CHECK-SAME: -> tensor<{{.*}}xui16, #ttnn.ttnn_layout<{{.*}} memref<1x128xui16, #ttnn.buffer_type<l1>>, <interleaved>>>

  // expert_scores: 32 x 4 = 128 -> memref<1x128xbf16, l1>
  // CHECK: "ttnn.to_layout"
  // CHECK-SAME: -> tensor<{{.*}}xbf16, #ttnn.ttnn_layout<{{.*}} memref<1x128xbf16, #ttnn.buffer_type<l1>>, <interleaved>>>
  func.func @dispatch_metadata_l1_canonical_form(
      %input:   tensor<1x1x32x128xbf16, #dram_input>,
      %indices: tensor<1x1x32x4xui16, #dram_idx>,
      %scores:  tensor<1x1x32x4xbf16, #dram_score>,
      %mapping: tensor<8x4xui16, #dram_map>
  ) -> (tensor<1x32x128xbf16, #dram_disp_out>,
        tensor<1x32x4xui16, #dram_idx_out>,
        tensor<1x32x4xbf16, #dram_sc_out>) {
    %disp, %idx, %sc = "ttnn.all_to_all_dispatch_metadata"(%input, %indices, %scores, %mapping)
        <{cluster_axis = 0 : i64, num_devices = 1 : i64}>
        : (tensor<1x1x32x128xbf16, #dram_input>,
           tensor<1x1x32x4xui16, #dram_idx>,
           tensor<1x1x32x4xbf16, #dram_score>,
           tensor<8x4xui16, #dram_map>)
        -> (tensor<1x32x128xbf16, #dram_disp_out>,
            tensor<1x32x4xui16, #dram_idx_out>,
            tensor<1x32x4xbf16, #dram_sc_out>)
    return %disp, %idx, %sc : tensor<1x32x128xbf16, #dram_disp_out>,
                              tensor<1x32x4xui16, #dram_idx_out>,
                              tensor<1x32x4xbf16, #dram_sc_out>
  }
}
