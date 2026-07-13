// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttnn-rm-layout-propagation %s | FileCheck %s

// Regression test: ChunkedScaledDotProductAttentionOp must keep its page_table
// and chunk_start_idx operands ROW_MAJOR through RowMajorLayoutPropagation.
//
// The tt-metal kernel hard-rejects a tiled page table
// ("TT_FATAL: Page table must be row major", sdpa_device_operation.cpp). The
// page_table / chunk_start_idx arguments arrive ROW_MAJOR. Because chunked SDPA
// now implements the op-model interface (it is no longer OpModelExempt),
// RowMajorLayoutPropagation queries the op model, learns the row-major operands
// are legal, and preserves them. Previously the op was OpModelExempt, so the
// pass could not query it and inserted a tilizing to_layout fixup on the
// page table (`opStopsRowMajorPropagation` -> `insertTiledFixup`), which then
// crashed at runtime.
//
// Query/key/value/output stay TILE (bf16 attention); only the integer
// page_table and chunk_start_idx are row-major.

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 104224, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1048640, dram_unreserved_end = 1073116480, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (4, 0), (5, 0), (0, 4), (3, 7), (1, 4), (7, 4), (6, 4), (2, 4), (4, 4), (5, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (4, 0), (5, 0), (0, 4), (3, 7), (1, 4), (7, 4), (6, 4), (2, 4), (4, 4), (5, 4)]}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<1536x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// page_table (%arg3) and chunk_start_idx (%arg4) arrive ROW_MAJOR.
// CHECK-DAG: #[[L2:.+]] = #ttnn.ttnn_layout<{{.*}}memref<1x4xsi32, #dram>
// CHECK-DAG: #[[L3:.+]] = #ttnn.ttnn_layout<{{.*}}memref<1x1xsi32, #dram>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xsi32, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>
module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @chunked_sdpa
  func.func @chunked_sdpa(%arg0: tensor<1x12x64x64xbf16, #ttnn_layout>, %arg1: tensor<128x12x32x64xbf16, #ttnn_layout1>, %arg2: tensor<128x12x32x64xbf16, #ttnn_layout1>, %arg3: tensor<1x4xi32, #ttnn_layout2>, %arg4: tensor<1xi32, #ttnn_layout3>) -> tensor<1x12x64x64xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{shape = #ttnn.shape<1x12x64x64>}> : (!ttnn.device) -> tensor<1x12x64x64xbf16, #ttnn_layout>
    // The page table / chunk_start_idx must NOT be tilized before the op.
    // CHECK-NOT: "ttnn.to_layout"(%arg3)
    // CHECK-NOT: "ttnn.to_layout"(%arg4)
    // The op consumes the row-major args directly.
    // CHECK: "ttnn.chunked_scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3, %arg4)
    // CHECK-SAME: tensor<1x4xi32, #[[L2]]>, tensor<1xi32, #[[L3]]>)
    %2 = "ttnn.chunked_scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3, %arg4) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xbf16, #ttnn_layout>, tensor<128x12x32x64xbf16, #ttnn_layout1>, tensor<128x12x32x64xbf16, #ttnn_layout1>, tensor<1x4xi32, #ttnn_layout2>, tensor<1xi32, #ttnn_layout3>) -> tensor<1x12x64x64xbf16, #ttnn_layout>
    return %2 : tensor<1x12x64x64xbf16, #ttnn_layout>
  }
}
