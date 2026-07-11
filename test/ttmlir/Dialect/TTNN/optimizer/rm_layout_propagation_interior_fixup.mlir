// REQUIRES: opmodel
// Regression for https://github.com/tenstorrent/tt-mlir/issues/8952
// RUN: ttmlir-opt --ttnn-rm-layout-propagation %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0, 1, 2, 3, 4, 5, 6, 7], [2 : i32, 4 : i32], [ 0x0x0x0,  0x0x0x1,  0x0x0x2,  0x0x0x3,  0x0x0x4,  0x0x0x5,  0x0x0x6,  0x0x0x7]>
#ttnn_layout_rm_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_tile_u32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 2x4, chipIds = [0, 1, 2, 3, 4, 5, 6, 7]>
  func.func @main(%arg0: tensor<1x1xsi32, #ttnn_layout_rm_si32> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x8xui32, #ttnn_layout_tile_u32> attributes {tt.function_type = "forward_device"} {
    // CHECK-LABEL: func.func @main
    // CHECK: %[[TC:.*]] = "ttnn.typecast"(%arg0)
    // CHECK: %[[FIX:.*]] = "ttnn.to_layout"(%[[TC]])
    // CHECK: "ttnn.all_gather"(%[[FIX]]) : (tensor<1x1xui32, #ttnn_layout_tile_u32>) -> tensor<1x8xui32, #ttnn_layout_tile_u32>
    %0 = "ttnn.to_layout"(%arg0) : (tensor<1x1xsi32, #ttnn_layout_rm_si32>) -> tensor<1x1xsi32, #ttnn_layout_tile_si32>
    %1 = "ttnn.typecast"(%0) : (tensor<1x1xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1xui32, #ttnn_layout_tile_u32>
    %2 = "ttnn.all_gather"(%1) <{all_gather_dim = 1 : si32, cluster_axis = 0 : ui32}> : (tensor<1x1xui32, #ttnn_layout_tile_u32>) -> tensor<1x8xui32, #ttnn_layout_tile_u32>
    return %2 : tensor<1x8xui32, #ttnn_layout_tile_u32>
  }
}
