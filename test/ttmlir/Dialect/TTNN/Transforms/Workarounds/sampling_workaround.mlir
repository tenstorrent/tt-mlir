// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
// Row-major layouts — workaround should convert index/param tensors to ROW_MAJOR
// (they typically arrive in TILE layout after prior ops).
#ttnn_layout_vals_tile  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_idx_tile   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_k_tile     = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout_p_tile     = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_out        = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<1xsi32, #dram>, <interleaved>>

module attributes {} {

  // Verify that index/param tensors in TILE layout are converted to ROW_MAJOR.
  // input_values stays in TILE (no workaround needed for values tensor).
  func.func @sampling_tile_inputs_get_row_major(
      %arg0: tensor<1x32xbf16, #ttnn_layout_vals_tile>,
      %arg1: tensor<1x32xi32, #ttnn_layout_idx_tile>,
      %arg2: tensor<1xui32, #ttnn_layout_k_tile>,
      %arg3: tensor<1xbf16, #ttnn_layout_p_tile>,
      %arg4: tensor<1xbf16, #ttnn_layout_p_tile>)
      -> tensor<1xi32, #ttnn_layout_out> {
    // CHECK-LABEL: func.func @sampling_tile_inputs_get_row_major
    // CHECK: %[[IDX:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: %[[K:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: %[[P:.*]] = "ttnn.to_layout"(%arg3)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: %[[T:.*]] = "ttnn.to_layout"(%arg4)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: "ttnn.sampling"(%arg0, %[[IDX]], %[[K]], %[[P]], %[[T]])
    %0 = "ttnn.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
        : (tensor<1x32xbf16, #ttnn_layout_vals_tile>,
           tensor<1x32xi32, #ttnn_layout_idx_tile>,
           tensor<1xui32, #ttnn_layout_k_tile>,
           tensor<1xbf16, #ttnn_layout_p_tile>,
           tensor<1xbf16, #ttnn_layout_p_tile>)
        -> tensor<1xi32, #ttnn_layout_out>
    return %0 : tensor<1xi32, #ttnn_layout_out>
  }

}
