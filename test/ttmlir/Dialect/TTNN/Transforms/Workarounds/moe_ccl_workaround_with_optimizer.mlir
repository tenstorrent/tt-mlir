// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --ttcore-register-device="mesh-shape=2,4" --ttnn-workaround="ttnn-optimization-level=1" --canonicalize -o %t %s --mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// Regression test: ensure that the workarounds for the sparse-MoE / CCL ops
// (all_to_all_dispatch, all_to_all_dispatch_metadata, all_to_all_combine,
// moe_expert_token_remap, and sparse_matmul) must remain in the workaround
// whitelist when the optimizer is enabled.

#dram = #ttnn.buffer_type<dram>

// all_to_all_dispatch: input -> RowMajor BFLOAT16, indices/mapping -> RowMajor
// UINT16, outputs cast back to the originally-requested encodings.
#disp_in    = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#disp_idx   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#disp_map   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#disp_out   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<8x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#disp_meta  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
module {
  func.func @all_to_all_dispatch(%input: tensor<1x1x128x2880xbf16, #disp_in>, %indices: tensor<1x1x128x4xsi32, #disp_idx>, %mapping: tensor<1x1x32x8xsi32, #disp_map>) -> (tensor<1x2x128x2880xbf16, #disp_out>, tensor<1x2x128x4xsi32, #disp_meta>) {
    // CHECK-LABEL: func.func @all_to_all_dispatch
    // input_tensor -> RowMajor BFLOAT16.
    // CHECK: %[[IN:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: -> tensor<1x1x128x2880xbf16, {{.*}} memref<128x2880xbf16,
    // expert_indices -> RowMajor UINT16.
    // CHECK: %[[IDX:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<1x1x128x4xui16, {{.*}} memref<128x4xui16,
    // expert_mapping -> RowMajor UINT16.
    // CHECK: %[[MAP:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: -> tensor<1x1x32x8xui16, {{.*}} memref<32x8xui16,
    // CHECK: "ttnn.all_to_all_dispatch"(%[[IN]], %[[IDX]], %[[MAP]])
    %dispatched, %metadata = "ttnn.all_to_all_dispatch"(%input, %indices, %mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64}> : (tensor<1x1x128x2880xbf16, #disp_in>, tensor<1x1x128x4xsi32, #disp_idx>, tensor<1x1x32x8xsi32, #disp_map>) -> (tensor<1x2x128x2880xbf16, #disp_out>, tensor<1x2x128x4xsi32, #disp_meta>)
    return %dispatched, %metadata : tensor<1x2x128x2880xbf16, #disp_out>, tensor<1x2x128x4xsi32, #disp_meta>
  }
}

// -----

// all_to_all_combine: input -> RowMajor BFLOAT16, metadata/mapping -> RowMajor
// UINT16, result cast back to the originally-requested encoding.
#dram = #ttnn.buffer_type<dram>
#comb_in   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<32x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#comb_meta = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#comb_map  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#comb_out  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<16x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @all_to_all_combine(%input: tensor<4x2x128x2880xbf16, #comb_in>, %metadata: tensor<1x2x128x4xsi32, #comb_meta>, %mapping: tensor<1x1x32x8xsi32, #comb_map>) -> tensor<4x1x128x2880xbf16, #comb_out> {
    // CHECK-LABEL: func.func @all_to_all_combine
    // input_tensor -> RowMajor BFLOAT16.
    // CHECK: %[[IN:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: -> tensor<4x2x128x2880xbf16, {{.*}} memref<1024x2880xbf16,
    // expert_metadata -> RowMajor UINT16.
    // CHECK: %[[META:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<1x2x128x4xui16, {{.*}} memref<256x4xui16,
    // expert_mapping -> RowMajor UINT16.
    // CHECK: %[[MAP:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: -> tensor<1x1x32x8xui16, {{.*}} memref<32x8xui16,
    // CHECK: "ttnn.all_to_all_combine"(%[[IN]], %[[META]], %[[MAP]])
    %result = "ttnn.all_to_all_combine"(%input, %metadata, %mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64, output_shard_dim = 1 : i64}> : (tensor<4x2x128x2880xbf16, #comb_in>, tensor<1x2x128x4xsi32, #comb_meta>, tensor<1x1x32x8xsi32, #comb_map>) -> tensor<4x1x128x2880xbf16, #comb_out>
    return %result : tensor<4x1x128x2880xbf16, #comb_out>
  }
}

// -----

// moe_expert_token_remap: topk_tensor -> RowMajor BFLOAT16, mapping/metadata ->
// RowMajor UINT16.
#dram = #ttnn.buffer_type<dram>
#rmp_topk = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#rmp_map  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#rmp_meta = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 128 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#rmp_red  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @moe_expert_token_remap(%topk: tensor<1x2x128x32xbf16, #rmp_topk>, %mapping: tensor<1x1x32x8xsi32, #rmp_map>, %metadata: tensor<1x2x128x4xsi32, #rmp_meta>) -> (tensor<1x2x128x4xbf16, #rmp_topk>, tensor<1x1x8x4xbf16, #rmp_red>) {
    // CHECK-LABEL: func.func @moe_expert_token_remap
    // topk_tensor -> RowMajor BFLOAT16.
    // CHECK: %[[TOPK:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: -> tensor<1x2x128x32xbf16, {{.*}} memref<256x32xbf16,
    // expert_mapping -> RowMajor UINT16.
    // CHECK: %[[MAP:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<1x1x32x8xui16, {{.*}} memref<32x8xui16,
    // expert_metadata -> RowMajor UINT16.
    // CHECK: %[[META:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: -> tensor<1x2x128x4xui16, {{.*}} memref<256x4xui16,
    // CHECK: "ttnn.moe_expert_token_remap"(%[[TOPK]], %[[MAP]], %[[META]])
    %mapping_out, %reduced = "ttnn.moe_expert_token_remap"(%topk, %mapping, %metadata) <{reduction_size = 32 : i64}> : (tensor<1x2x128x32xbf16, #rmp_topk>, tensor<1x1x32x8xsi32, #rmp_map>, tensor<1x2x128x4xsi32, #rmp_meta>) -> (tensor<1x2x128x4xbf16, #rmp_topk>, tensor<1x1x8x4xbf16, #rmp_red>)
    return %mapping_out, %reduced : tensor<1x2x128x4xbf16, #rmp_topk>, tensor<1x1x8x4xbf16, #rmp_red>
  }
}

// -----

// all_to_all_dispatch_metadata: input/indices/scores -> L1-interleaved RowMajor
// (BFLOAT16 for activations, UINT16 for indices), kernel sharded outputs cast
// back to the originally-requested DRAM encodings.
#dram = #ttnn.buffer_type<dram>
#dm_in    = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x128xbf16, #dram>, <interleaved>>
#dm_idx   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x4xui16, #dram>, <interleaved>>
#dm_score = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x4xbf16, #dram>, <interleaved>>
#dm_map   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x4xui16, #dram>, <interleaved>>
#dm_disp  = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x128xbf16, #dram>, <interleaved>>
#dm_iout  = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x4xui16, #dram>, <interleaved>>
#dm_sout  = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x4xbf16, #dram>, <interleaved>>
module {
  func.func @all_to_all_dispatch_metadata(%input: tensor<1x1x32x128xbf16, #dm_in>, %indices: tensor<1x1x32x4xui16, #dm_idx>, %scores: tensor<1x1x32x4xbf16, #dm_score>, %mapping: tensor<8x4xui16, #dm_map>) -> (tensor<1x32x128xbf16, #dm_disp>, tensor<1x32x4xui16, #dm_iout>, tensor<1x32x4xbf16, #dm_sout>) {
    // CHECK-LABEL: func.func @all_to_all_dispatch_metadata
    // input_tensor -> L1-interleaved RowMajor BFLOAT16.
    // CHECK: %[[IN:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: -> tensor<1x1x32x128xbf16, {{.*}} memref<1x4096xbf16, #ttnn.buffer_type<l1>>
    // expert_indices -> L1-interleaved RowMajor UINT16.
    // CHECK: %[[IDX:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<1x1x32x4xui16, {{.*}} memref<1x128xui16, #ttnn.buffer_type<l1>>
    // expert_scores -> L1-interleaved RowMajor BFLOAT16.
    // CHECK: %[[SC:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: -> tensor<1x1x32x4xbf16, {{.*}} memref<1x128xbf16, #ttnn.buffer_type<l1>>
    // CHECK: "ttnn.all_to_all_dispatch_metadata"(%[[IN]], %[[IDX]], %[[SC]], %arg3)
    %disp, %idx, %sc = "ttnn.all_to_all_dispatch_metadata"(%input, %indices, %scores, %mapping) <{cluster_axis = 0 : i64, num_devices = 1 : i64}> : (tensor<1x1x32x128xbf16, #dm_in>, tensor<1x1x32x4xui16, #dm_idx>, tensor<1x1x32x4xbf16, #dm_score>, tensor<8x4xui16, #dm_map>) -> (tensor<1x32x128xbf16, #dm_disp>, tensor<1x32x4xui16, #dm_iout>, tensor<1x32x4xbf16, #dm_sout>)
    return %disp, %idx, %sc : tensor<1x32x128xbf16, #dm_disp>, tensor<1x32x4xui16, #dm_iout>, tensor<1x32x4xbf16, #dm_sout>
  }
}

// -----

// sparse_matmul: sparsity (3rd operand) must be forced to RowMajor BFLOAT16 (a
// f32 sparsity makes the kernel's uint16 reinterpret read 0x0000 = invalid and
// silently zero expert outputs; a UInt16 sparsity, e.g. the moe_expert_token_remap
// reduced output, must likewise become BFloat16 so the kernel sees the right tag).
#dram = #ttnn.buffer_type<dram>
#sm_a  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#sm_b  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 256 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#sm_sf = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#sm_su = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>
#sm_o  = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0 * 512 + d1 * 128 + d2 * 128 + d3 * 32 + d4, d5), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func public @sparse_matmul_sparsity_f32_to_bf16(%a: tensor<1x4x32x256xbf16, #sm_a>, %b: tensor<1x4x256x64xbf16, #sm_b>, %s: tensor<1x4x1x4xf32, #sm_sf>) -> tensor<1x4x1x4x32x64xbf16, #sm_o> {
    // CHECK-LABEL: func.func public @sparse_matmul_sparsity_f32_to_bf16
    // CHECK: %[[SPARSITY:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: tensor<1x4x1x4xf32,
    // CHECK-SAME: -> tensor<1x4x1x4xbf16, {{.*}} memref<4x4xbf16,
    // CHECK: "ttnn.sparse_matmul"(%arg0, %arg1, %[[SPARSITY]])
    %0 = "ttnn.sparse_matmul"(%a, %b, %s) <{is_input_a_sparse = false, is_input_b_sparse = true}> : (tensor<1x4x32x256xbf16, #sm_a>, tensor<1x4x256x64xbf16, #sm_b>, tensor<1x4x1x4xf32, #sm_sf>) -> tensor<1x4x1x4x32x64xbf16, #sm_o>
    return %0 : tensor<1x4x1x4x32x64xbf16, #sm_o>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#sm_a  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#sm_b  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 256 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#sm_su = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>
#sm_o  = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0 * 512 + d1 * 128 + d2 * 128 + d3 * 32 + d4, d5), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func public @sparse_matmul_sparsity_ui16_to_bf16(%a: tensor<1x4x32x256xbf16, #sm_a>, %b: tensor<1x4x256x64xbf16, #sm_b>, %s: tensor<1x4x1x4xui16, #sm_su>) -> tensor<1x4x1x4x32x64xbf16, #sm_o> {
    // CHECK-LABEL: func.func public @sparse_matmul_sparsity_ui16_to_bf16
    // CHECK: %[[SPARSITY:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: tensor<1x4x1x4xui16,
    // CHECK-SAME: -> tensor<1x4x1x4xbf16, {{.*}} memref<4x4xbf16,
    // CHECK: "ttnn.sparse_matmul"(%arg0, %arg1, %[[SPARSITY]])
    %0 = "ttnn.sparse_matmul"(%a, %b, %s) <{is_input_a_sparse = false, is_input_b_sparse = true}> : (tensor<1x4x32x256xbf16, #sm_a>, tensor<1x4x256x64xbf16, #sm_b>, tensor<1x4x1x4xui16, #sm_su>) -> tensor<1x4x1x4x32x64xbf16, #sm_o>
    return %0 : tensor<1x4x1x4x32x64xbf16, #sm_o>
  }
}
