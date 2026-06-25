// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
// Row-major layouts — workaround should convert index/param tensors to ROW_MAJOR
// (they typically arrive in TILE layout after prior ops).
#ttnn_layout_vals_tile  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_idx_tile   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_k_tile     = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout_k_si32_rm  = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>
#ttnn_layout_p_tile     = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_out        = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>

module attributes {} {

  // The workaround pass is expected to:
  //   1. SamplingOpRank2RewritePattern: rank-2 input -> reshape to rank-4,
  //      rank-4 ttnn.sampling, reshape result back to rank-1.
  //   2. Operand workarounds: index/param tensors arriving in TILE get a
  //      ttnn.to_layout to ROW_MAJOR; the kernel produces ui32, so the result
  //      is retyped and a conversion back to si32 is inserted for the consumer.
  func.func @sampling_rank2_input_decomposes_to_rank4(
      %arg0: tensor<1x32xbf16, #ttnn_layout_vals_tile>,
      %arg1: tensor<1x32xi32, #ttnn_layout_idx_tile>,
      %arg2: tensor<1xui32, #ttnn_layout_k_tile>,
      %arg3: tensor<1xbf16, #ttnn_layout_p_tile>,
      %arg4: tensor<1xbf16, #ttnn_layout_p_tile>)
      -> tensor<1xsi32, #ttnn_layout_out> {
    // CHECK-LABEL: func.func @sampling_rank2_input_decomposes_to_rank4
    // CHECK-DAG: "ttnn.reshape"
    // CHECK-DAG-SAME: shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]
    // CHECK-DAG: "ttnn.to_layout"{{.*}}memref<{{[0-9x]+}}si32,
    // CHECK-DAG: "ttnn.to_layout"{{.*}}memref<{{[0-9x]+}}ui32,
    // CHECK-DAG: "ttnn.to_layout"{{.*}}memref<{{[0-9x]+}}bf16,
    // CHECK-DAG: "ttnn.to_layout"{{.*}}memref<{{[0-9x]+}}bf16,
    // ttnn.sampling now operates on the kernel-true rank-4 shape and yields ui32.
    // CHECK: "ttnn.sampling"
    // CHECK-SAME: (tensor<1x1x1x32xbf16{{.*}}>, tensor<1x1x1x32xsi32{{.*}}>, tensor<1xui32{{.*}}>, tensor<1xbf16{{.*}}>, tensor<1xbf16{{.*}}>)
    // CHECK-SAME: -> tensor<1x1x1x1xui32
    // Result is converted from ui32 -> si32 (dtype workaround) and reshaped
    // from [1,1,1,batch] -> [batch] (rank-2 decomposition's trailing reshape).
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: -> tensor<1x1x1x1xsi32
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32]
    %0 = "ttnn.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
        : (tensor<1x32xbf16, #ttnn_layout_vals_tile>,
           tensor<1x32xi32, #ttnn_layout_idx_tile>,
           tensor<1xui32, #ttnn_layout_k_tile>,
           tensor<1xbf16, #ttnn_layout_p_tile>,
           tensor<1xbf16, #ttnn_layout_p_tile>)
        -> tensor<1xsi32, #ttnn_layout_out>
    return %0 : tensor<1xsi32, #ttnn_layout_out>
  }

  // The ttnn::sampling kernel requires k as UINT32. When k arrives as a
  // different dtype (here: SI32), the operand workaround must produce a
  // UINT32 k before the sampling op (the workaround pass folds the layout
  // and dtype change into a single ttnn.to_layout). The runtime handler
  // used to do this implicitly; expressing it as IR keeps EmitPy / EmitC
  // codegen paths correct.
  func.func @sampling_k_si32_gets_retyped_to_ui32(
      %arg0: tensor<1x32xbf16, #ttnn_layout_vals_tile>,
      %arg1: tensor<1x32xi32, #ttnn_layout_idx_tile>,
      %arg2: tensor<1xsi32, #ttnn_layout_k_si32_rm>,
      %arg3: tensor<1xbf16, #ttnn_layout_p_tile>,
      %arg4: tensor<1xbf16, #ttnn_layout_p_tile>)
      -> tensor<1xsi32, #ttnn_layout_out> {
    // CHECK-LABEL: func.func @sampling_k_si32_gets_retyped_to_ui32
    // CHECK: "ttnn.to_layout"(%arg2)
    // CHECK-SAME: (tensor<1xsi32{{.*}}>) -> tensor<1xui32
    // CHECK: "ttnn.sampling"
    %0 = "ttnn.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
        : (tensor<1x32xbf16, #ttnn_layout_vals_tile>,
           tensor<1x32xi32, #ttnn_layout_idx_tile>,
           tensor<1xsi32, #ttnn_layout_k_si32_rm>,
           tensor<1xbf16, #ttnn_layout_p_tile>,
           tensor<1xbf16, #ttnn_layout_p_tile>)
        -> tensor<1xsi32, #ttnn_layout_out>
    return %0 : tensor<1xsi32, #ttnn_layout_out>
  }

}
