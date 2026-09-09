// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

#dram = #ttnn.buffer_type<dram>

#cache_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3),
                                   <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>,
                                   <interleaved>>
#input_layout_b4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3),
                                     <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>,
                                     <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1),
                               <1x1>, memref<1x2xi32, #dram>, <interleaved>>
#batch_idx_layout_2 = #ttnn.ttnn_layout<(d0) -> (d0),
                                        <1x1>, memref<2xi32, #dram>, <interleaved>>
#batch_idx_layout_4 = #ttnn.ttnn_layout<(d0) -> (d0),
                                        <1x1>, memref<4xi32, #dram>, <interleaved>>

// Happy path: input_batch == batch_idx_tensor.shape[0] > 1. Should verify.
module {
  func.func @paged_fill_cache_ttnn_batched_ok(
      %cache: tensor<1x32x64x128xbf16, #cache_layout> {ttcore.kv_cache},
      %input: tensor<4x32x1x128xbf16, #input_layout_b4>,
      %page_table: tensor<1x2xi32, #pt_layout>,
      %batch_idx: tensor<4xi32, #batch_idx_layout_4>) {
    "ttnn.paged_fill_cache"(%cache, %input, %page_table, %batch_idx) : (
        tensor<1x32x64x128xbf16, #cache_layout>,
        tensor<4x32x1x128xbf16, #input_layout_b4>,
        tensor<1x2xi32, #pt_layout>,
        tensor<4xi32, #batch_idx_layout_4>) -> ()
    return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#cache_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3),
                                   <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>,
                                   <interleaved>>
#input_layout_b4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3),
                                     <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>,
                                     <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1),
                               <1x1>, memref<1x2xi32, #dram>, <interleaved>>
#batch_idx_layout_2 = #ttnn.ttnn_layout<(d0) -> (d0),
                                        <1x1>, memref<2xi32, #dram>, <interleaved>>

// batch_idx_tensor.shape[0] != input batch (4 != 2).
module {
  func.func @paged_fill_cache_ttnn_batch_idx_size_mismatch(
      %cache: tensor<1x32x64x128xbf16, #cache_layout> {ttcore.kv_cache},
      %input: tensor<4x32x1x128xbf16, #input_layout_b4>,
      %page_table: tensor<1x2xi32, #pt_layout>,
      %batch_idx: tensor<2xi32, #batch_idx_layout_2>) {
    // expected-error @+1 {{'ttnn.paged_fill_cache' op Batch index tensor must have dim 0 equal to input batch (4), got 2}}
    "ttnn.paged_fill_cache"(%cache, %input, %page_table, %batch_idx) : (
        tensor<1x32x64x128xbf16, #cache_layout>,
        tensor<4x32x1x128xbf16, #input_layout_b4>,
        tensor<1x2xi32, #pt_layout>,
        tensor<2xi32, #batch_idx_layout_2>) -> ()
    return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#cache_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3),
                                   <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>,
                                   <interleaved>>
#input_layout_b4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3),
                                     <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>,
                                     <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1),
                               <1x1>, memref<1x2xi32, #dram>, <interleaved>>

// input_batch > 1 without batch_idx_tensor must be rejected.
module {
  func.func @paged_fill_cache_ttnn_multi_batch_without_idx_tensor(
      %cache: tensor<1x32x64x128xbf16, #cache_layout> {ttcore.kv_cache},
      %input: tensor<4x32x1x128xbf16, #input_layout_b4>,
      %page_table: tensor<1x2xi32, #pt_layout>) {
    // expected-error @+1 {{'ttnn.paged_fill_cache' op Input batch must be statically 1 when no batch_idx_tensor is provided, got 4}}
    "ttnn.paged_fill_cache"(%cache, %input, %page_table) : (
        tensor<1x32x64x128xbf16, #cache_layout>,
        tensor<4x32x1x128xbf16, #input_layout_b4>,
        tensor<1x2xi32, #pt_layout>) -> ()
    return
  }
}
