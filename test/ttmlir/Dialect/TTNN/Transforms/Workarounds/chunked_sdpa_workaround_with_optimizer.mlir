// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Regression test: ChunkedScaledDotProductAttentionOp must remain in the
// workaround allowlist when the optimizer is enabled.
//
// The TTNN workarounds pass restricts its operand layout/dtype workarounds to
// a small allowlist when the optimizer runs (ttnn-optimization-level >= 1).
// ChunkedScaledDotProductAttentionOp was absent from that set, so its
// createChunkedScaledDotProductAttentionOpOperandsWorkarounds (which forces the
// page_table and chunk_start_idx operands to ROW_MAJOR) was skipped at opt >= 1.
// Without it, the optimizer's layout propagation leaves the page table TILE and
// the tt-metal kernel aborts at runtime with:
//   "TT_FATAL: Page table must be row major".

// page_table (%arg3) and chunk_start_idx (%arg4) must be coerced to ROW_MAJOR
// even with the optimizer enabled.
// CHECK-DAG: #[[PAGE_TABLE_RM_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x4xsi32, #dram>
// CHECK-DAG: #[[CHUNK_START_RM_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x1xsi32, #dram>
// CHECK-DAG: %[[PAGE_TABLE:.*]] = "ttnn.to_layout"(%arg3){{.*}} -> tensor<1x4xsi32, #[[PAGE_TABLE_RM_LAYOUT]]>
// CHECK-DAG: %[[CHUNK_START:.*]] = "ttnn.to_layout"(%arg4){{.*}} -> tensor<1xsi32, #[[CHUNK_START_RM_LAYOUT]]>
// CHECK: "ttnn.chunked_scaled_dot_product_attention"
// CHECK-SAME: %[[PAGE_TABLE]], %[[CHUNK_START]]
module attributes {} {
  // query: [num_users, num_heads, chunk_len, head_size]; key/value: paged cache
  // [num_blocks, num_kv_heads, block_size, head_size].
  func.func @chunked_sdpa(%arg0: tensor<1x12x64x64xf32>, %arg1: tensor<128x12x32x64xf32>, %arg2: tensor<128x12x32x64xf32>, %arg3: tensor<1x4xi32>, %arg4: tensor<1xi32>) -> tensor<1x12x64x64xf32> {
    %0 = ttir.empty() : tensor<1x12x64x64xf32>
    %1 = "ttir.chunked_scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1xi32>, tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32>
    return %1 : tensor<1x12x64x64xf32>
  }
}
