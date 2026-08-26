#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 105536, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2968640, dram_unreserved_end = 1073108608, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (4, 0), (1, 2), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 4), (4, 4), (3, 6)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (4, 0), (1, 2), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 4), (4, 4), (3, 6)]}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4008x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<256x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x256x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x512xsi32, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 511 + d1, d2), <1x1>, memref<511x128256xbf16, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 511 + d1, d2), <1x1>, memref<511x1xf32, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x4008x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x64x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <1x1>, memref<512x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout19 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <1x1>, memref<512x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout20 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x1>, memref<16x256x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout21 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 32 + d2 * 32 + d3, d4), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout22 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout23 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout24 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout25 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout26 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x512xui32, #dram>, <interleaved>>
#ttnn_layout27 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128256x2048xbf16, #dram>, <interleaved>>
#ttnn_layout28 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <8x8>, memref<1x16x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout29 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout30 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x8>, memref<1x2x!ttcore.tile<32x32, si32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout31 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x512xbf16, #dram>, <interleaved>>
#ttnn_layout33 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x8>, memref<1x2x!ttcore.tile<32x32, si32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout34 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, si32>, #l1>, <interleaved>>
#ttnn_layout35 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x8>, memref<1x2x!ttcore.tile<32x32, si32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout36 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <8x8>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout37 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout38 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout39 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout40 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x16>, memref<1x1x!ttcore.tile<32x32, si32>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout41 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x16>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout42 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x16>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout43 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout44 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x8>, memref<1x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout45 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout46 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x16>, memref<1x1x!ttcore.tile<32x32, si32>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout47 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x16>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout48 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x16>, memref<1x1x!ttcore.tile<32x32, u32>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout49 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout50 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout51 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout52 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout53 = #ttnn.ttnn_layout<(d0) -> (0, d0), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout54 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <8x8>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout55 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x8>, memref<1x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,0)>]>>
#ttnn_layout56 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout57 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout58 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x16x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout59 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <8x8>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
#ttnn_layout60 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <16x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout61 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <16x1>, memref<1x2x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout62 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <16x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout63 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x64>, memref<16x1x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout64 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <1x64>, memref<16x1x!ttcore.tile<32x32, f32>, #l1>, <width_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout65 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <8x8>, memref<2x8x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout66 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <8x8>, memref<2x1x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout67 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <8x8>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout68 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<96x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout69 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <8x8>, memref<2x12x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout70 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <8x8>, memref<1x24x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout71 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <8x8>, memref<1x16x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout72 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 512 + d2, d3), <8x8>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout73 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <16x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout74 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4096 + d1 * 512 + d2 * 512 + d3, d4), <8x8>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout75 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <64x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout76 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <64x1>, memref<8x2x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout77 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4096 + d1 * 512 + d2 * 512 + d3, d4), <64x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout78 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4096 + d1 * 512 + d2 * 512 + d3, d4), <64x1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout79 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 2048 + d2 * 512 + d3, d4), <64x1>, memref<8x2x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout80 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 32 + d2 * 32 + d3, d4), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (0,0)>]>>
#ttnn_layout81 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <8x8>, memref<1x16x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
#ttnn_layout82 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <8x8>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout83 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout84 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 512 + d2, d3), <64x1>, memref<8x16x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout85 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <8x8>, memref<2x32x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,7)>]>>
#ttnn_layout86 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 512 + d1, d2), <16x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
#ttnn_layout87 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <16x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <height_sharded>, core_ranges = <[#ttnn.core_range<(0,0), (7,1)>]>>
module @llama_3_2_1b_512_fwd attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @llama_3_2_1b_512_fwd attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0], meshTopology = [linear, linear]>
      func.func @main(%arg0: tensor<128256x2048xbf16, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<128256x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.embed_tokens.weight"}, %arg1: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.q_proj.weight"}, %arg2: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.k_proj.weight"}, %arg3: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.v_proj.weight"}, %arg4: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.self_attn.o_proj.weight"}, %arg5: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.gate_proj.weight"}, %arg6: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.up_proj.weight"}, %arg7: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.mlp.down_proj.weight"}, %arg8: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.input_layernorm.weight"}, %arg9: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.0.post_attention_layernorm.weight"}, %arg10: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.q_proj.weight"}, %arg11: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.k_proj.weight"}, %arg12: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.v_proj.weight"}, %arg13: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.self_attn.o_proj.weight"}, %arg14: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.gate_proj.weight"}, %arg15: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.up_proj.weight"}, %arg16: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.mlp.down_proj.weight"}, %arg17: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.input_layernorm.weight"}, %arg18: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.1.post_attention_layernorm.weight"}, %arg19: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.q_proj.weight"}, %arg20: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.k_proj.weight"}, %arg21: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.v_proj.weight"}, %arg22: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.self_attn.o_proj.weight"}, %arg23: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.gate_proj.weight"}, %arg24: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.up_proj.weight"}, %arg25: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.mlp.down_proj.weight"}, %arg26: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.input_layernorm.weight"}, %arg27: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.2.post_attention_layernorm.weight"}, %arg28: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.q_proj.weight"}, %arg29: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.k_proj.weight"}, %arg30: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.v_proj.weight"}, %arg31: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.self_attn.o_proj.weight"}, %arg32: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.gate_proj.weight"}, %arg33: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.up_proj.weight"}, %arg34: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.mlp.down_proj.weight"}, %arg35: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.input_layernorm.weight"}, %arg36: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.3.post_attention_layernorm.weight"}, %arg37: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.q_proj.weight"}, %arg38: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.k_proj.weight"}, %arg39: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.v_proj.weight"}, %arg40: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.self_attn.o_proj.weight"}, %arg41: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.gate_proj.weight"}, %arg42: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.up_proj.weight"}, %arg43: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.mlp.down_proj.weight"}, %arg44: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.input_layernorm.weight"}, %arg45: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.4.post_attention_layernorm.weight"}, %arg46: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.q_proj.weight"}, %arg47: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.k_proj.weight"}, %arg48: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.v_proj.weight"}, %arg49: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.self_attn.o_proj.weight"}, %arg50: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.gate_proj.weight"}, %arg51: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.up_proj.weight"}, %arg52: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.mlp.down_proj.weight"}, %arg53: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.input_layernorm.weight"}, %arg54: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.5.post_attention_layernorm.weight"}, %arg55: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.q_proj.weight"}, %arg56: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.k_proj.weight"}, %arg57: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.v_proj.weight"}, %arg58: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.self_attn.o_proj.weight"}, %arg59: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.gate_proj.weight"}, %arg60: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.up_proj.weight"}, %arg61: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.mlp.down_proj.weight"}, %arg62: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.input_layernorm.weight"}, %arg63: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.6.post_attention_layernorm.weight"}, %arg64: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.q_proj.weight"}, %arg65: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.k_proj.weight"}, %arg66: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.v_proj.weight"}, %arg67: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.self_attn.o_proj.weight"}, %arg68: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.gate_proj.weight"}, %arg69: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.up_proj.weight"}, %arg70: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.mlp.down_proj.weight"}, %arg71: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.input_layernorm.weight"}, %arg72: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.7.post_attention_layernorm.weight"}, %arg73: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.q_proj.weight"}, %arg74: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.k_proj.weight"}, %arg75: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.v_proj.weight"}, %arg76: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.self_attn.o_proj.weight"}, %arg77: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.gate_proj.weight"}, %arg78: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.up_proj.weight"}, %arg79: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.mlp.down_proj.weight"}, %arg80: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.input_layernorm.weight"}, %arg81: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.8.post_attention_layernorm.weight"}, %arg82: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.q_proj.weight"}, %arg83: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.k_proj.weight"}, %arg84: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.v_proj.weight"}, %arg85: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.self_attn.o_proj.weight"}, %arg86: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.gate_proj.weight"}, %arg87: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.up_proj.weight"}, %arg88: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.mlp.down_proj.weight"}, %arg89: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.input_layernorm.weight"}, %arg90: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.9.post_attention_layernorm.weight"}, %arg91: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.q_proj.weight"}, %arg92: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.k_proj.weight"}, %arg93: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.v_proj.weight"}, %arg94: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.self_attn.o_proj.weight"}, %arg95: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.gate_proj.weight"}, %arg96: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.up_proj.weight"}, %arg97: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.mlp.down_proj.weight"}, %arg98: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.input_layernorm.weight"}, %arg99: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.10.post_attention_layernorm.weight"}, %arg100: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.q_proj.weight"}, %arg101: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.k_proj.weight"}, %arg102: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.v_proj.weight"}, %arg103: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.self_attn.o_proj.weight"}, %arg104: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.gate_proj.weight"}, %arg105: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.up_proj.weight"}, %arg106: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.mlp.down_proj.weight"}, %arg107: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.input_layernorm.weight"}, %arg108: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.11.post_attention_layernorm.weight"}, %arg109: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.q_proj.weight"}, %arg110: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.k_proj.weight"}, %arg111: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.v_proj.weight"}, %arg112: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.self_attn.o_proj.weight"}, %arg113: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.gate_proj.weight"}, %arg114: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.up_proj.weight"}, %arg115: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.mlp.down_proj.weight"}, %arg116: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.input_layernorm.weight"}, %arg117: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.12.post_attention_layernorm.weight"}, %arg118: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.q_proj.weight"}, %arg119: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.k_proj.weight"}, %arg120: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.v_proj.weight"}, %arg121: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.self_attn.o_proj.weight"}, %arg122: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.gate_proj.weight"}, %arg123: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.up_proj.weight"}, %arg124: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.mlp.down_proj.weight"}, %arg125: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.input_layernorm.weight"}, %arg126: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.13.post_attention_layernorm.weight"}, %arg127: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.q_proj.weight"}, %arg128: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.k_proj.weight"}, %arg129: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.v_proj.weight"}, %arg130: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.self_attn.o_proj.weight"}, %arg131: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.gate_proj.weight"}, %arg132: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.up_proj.weight"}, %arg133: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.mlp.down_proj.weight"}, %arg134: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.input_layernorm.weight"}, %arg135: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.14.post_attention_layernorm.weight"}, %arg136: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.q_proj.weight"}, %arg137: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.k_proj.weight"}, %arg138: tensor<512x2048xbf16, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<512x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.v_proj.weight"}, %arg139: tensor<2048x2048xbf16, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.self_attn.o_proj.weight"}, %arg140: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.gate_proj.weight"}, %arg141: tensor<8192x2048xbf16, #ttnn_layout3> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<8192x2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.up_proj.weight"}, %arg142: tensor<2048x8192xbf16, #ttnn_layout4> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048x8192xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.mlp.down_proj.weight"}, %arg143: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.input_layernorm.weight"}, %arg144: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.layers.15.post_attention_layernorm.weight"}, %arg145: tensor<2048xbf16, #ttnn_layout5> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<2048xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.norm.weight"}, %arg146: tensor<32xf32, #ttnn_layout6> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<32xf32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "model.rotary_emb.inv_freq"}, %arg147: tensor<1x512xsi32, #ttnn_layout7> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x512xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "input_ids"}, %arg148: tensor<1x512xsi32, #ttnn_layout7> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x512xi64>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "attention_mask"}, %arg149: tensor<1x511x128256xbf16, #ttnn_layout8> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x511x128256xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "labels_one_hot"}, %arg150: tensor<1x511x1xf32, #ttnn_layout9> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x511x1xf32>>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "loss_weight"}) -> (tensor<1x1x1xf32, #ttnn_layout10>, tensor<128256x2048xbf16, #ttnn_layout>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<1x512xsi32, #ttnn_layout11>, tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x511x1xf32, #ttnn_layout13>) attributes {tt.function_type = "forward_device"} {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.full"(%0) <{fill_value = 0.353553385 : f32, shape = #ttnn.shape<1x1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1x1xf32, #ttnn_layout21>
        %2 = "ttnn.full"(%0) <{fill_value = 512 : i32, shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        %3 = "ttnn.full"(%0) <{fill_value = 0.353553385 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        %4 = "ttnn.full"(%0) <{fill_value = 0xFF800000 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xbf16, #ttnn_layout24>
        %5 = "ttnn.zeros"(%0) <{shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xbf16, #ttnn_layout24>
        %6 = "ttnn.full"(%0) <{fill_value = 9.99999974E-6 : f32, shape = #ttnn.shape<1x1x1>}> : (!ttnn.device) -> tensor<1x1x1xf32, #ttnn_layout10>
        %7 = "ttnn.ones"(%0) <{shape = #ttnn.shape<1x1x1>}> : (!ttnn.device) -> tensor<1x1x1xf32, #ttnn_layout10>
        %8 = "ttnn.zeros"(%0) <{shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        %9 = "ttnn.ones"(%0) <{shape = #ttnn.shape<1x1x1x1>}> : (!ttnn.device) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        %10 = "ttnn.zeros"(%0) <{shape = #ttnn.shape<1>}> : (!ttnn.device) -> tensor<1xsi32, #ttnn_layout25>
        %11 = "ttnn.typecast"(%arg147) : (tensor<1x512xsi32, #ttnn_layout7>) -> tensor<1x512xui32, #ttnn_layout26>
        %12 = "ttnn.to_layout"(%arg0) : (tensor<128256x2048xbf16, #ttnn_layout>) -> tensor<128256x2048xbf16, #ttnn_layout27>
        %13 = "ttnn.embedding"(%11, %12) : (tensor<1x512xui32, #ttnn_layout26>, tensor<128256x2048xbf16, #ttnn_layout27>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<128256x2048xbf16, #ttnn_layout27>) -> ()
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1x512xui32, #ttnn_layout26>) -> ()
        %14 = "ttnn.to_memory_config"(%13) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %15 = "ttnn.arange"(%0) <{end = 512 : si64, start = 0 : si64, step = 1 : si64}> : (!ttnn.device) -> tensor<512xsi32, #ttnn_layout29>
        %16 = "ttnn.to_memory_config"(%15) : (tensor<512xsi32, #ttnn_layout29>) -> tensor<512xsi32, #ttnn_layout30>
        "ttnn.deallocate"(%15) <{force = false}> : (tensor<512xsi32, #ttnn_layout29>) -> ()
        %17 = "ttnn.to_memory_config"(%10) : (tensor<1xsi32, #ttnn_layout25>) -> tensor<1xsi32, #ttnn_layout31>
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1xsi32, #ttnn_layout25>) -> ()
        %18 = "ttnn.add"(%16, %17) : (tensor<512xsi32, #ttnn_layout30>, tensor<1xsi32, #ttnn_layout31>) -> tensor<512xsi32, #ttnn_layout30>
        "ttnn.deallocate"(%17) <{force = false}> : (tensor<1xsi32, #ttnn_layout31>) -> ()
        "ttnn.deallocate"(%16) <{force = false}> : (tensor<512xsi32, #ttnn_layout30>) -> ()
        %19 = "ttnn.typecast"(%arg148) : (tensor<1x512xsi32, #ttnn_layout7>) -> tensor<1x512xbf16, #ttnn_layout32>
        "ttnn.deallocate"(%arg148) <{force = false}> : (tensor<1x512xsi32, #ttnn_layout7>) -> ()
        %20 = "ttnn.arange"(%0) <{end = 1 : si64, start = 0 : si64, step = 1 : si64}> : (!ttnn.device) -> tensor<1xsi32, #ttnn_layout25>
        %21 = "ttnn.reshape"(%20) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xsi32, #ttnn_layout25>) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        "ttnn.deallocate"(%20) <{force = false}> : (tensor<1xsi32, #ttnn_layout25>) -> ()
        %22 = "ttnn.reshape"(%18) <{shape = [1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xsi32, #ttnn_layout30>) -> tensor<1x1x512xsi32, #ttnn_layout33>
        %23 = "ttnn.reshape"(%18) <{shape = [1 : i32, 1 : i32, 512 : i32, 1 : i32]}> : (tensor<512xsi32, #ttnn_layout30>) -> tensor<1x1x512x1xsi32, #ttnn_layout34>
        %24 = "ttnn.reshape"(%18) <{shape = [1 : i32, 1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xsi32, #ttnn_layout30>) -> tensor<1x1x1x512xsi32, #ttnn_layout35>
        "ttnn.deallocate"(%18) <{force = false}> : (tensor<512xsi32, #ttnn_layout30>) -> ()
        %25 = "ttnn.ge"(%23, %24) : (tensor<1x1x512x1xsi32, #ttnn_layout34>, tensor<1x1x1x512xsi32, #ttnn_layout35>) -> tensor<1x1x512x512xbf16, #ttnn_layout36>
        "ttnn.deallocate"(%23) <{force = false}> : (tensor<1x1x512x1xsi32, #ttnn_layout34>) -> ()
        %26 = "ttnn.to_memory_config"(%21) : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xsi32, #ttnn_layout37>
        %27 = "ttnn.to_memory_config"(%9) : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xsi32, #ttnn_layout37>
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %28 = "ttnn.add"(%26, %27) : (tensor<1x1x1x1xsi32, #ttnn_layout37>, tensor<1x1x1x1xsi32, #ttnn_layout37>) -> tensor<1x1x1x1xsi32, #ttnn_layout37>
        "ttnn.deallocate"(%27) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> ()
        %29 = "ttnn.to_memory_config"(%8) : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xsi32, #ttnn_layout37>
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %30 = "ttnn.gt"(%29, %26) : (tensor<1x1x1x1xsi32, #ttnn_layout37>, tensor<1x1x1x1xsi32, #ttnn_layout37>) -> tensor<1x1x1x1xbf16, #ttnn_layout38>
        "ttnn.deallocate"(%26) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> ()
        %31 = "ttnn.typecast"(%30) : (tensor<1x1x1x1xbf16, #ttnn_layout38>) -> tensor<1x1x1x1xf32, #ttnn_layout39>
        "ttnn.deallocate"(%30) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout38>) -> ()
        %32 = "ttnn.to_memory_config"(%31) : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        "ttnn.deallocate"(%31) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> ()
        %33 = "ttnn.typecast"(%28) : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> tensor<1x1x1x1xf32, #ttnn_layout39>
        "ttnn.deallocate"(%28) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> ()
        %34 = "ttnn.to_memory_config"(%33) : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        "ttnn.deallocate"(%33) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> ()
        %35 = "ttnn.typecast"(%21) : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        "ttnn.deallocate"(%21) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %36 = "ttnn.to_memory_config"(%32) : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x1x1x1xf32, #ttnn_layout39>
        "ttnn.deallocate"(%32) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        %37 = "ttnn.to_memory_config"(%35) : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x1x1x1xf32, #ttnn_layout39>
        "ttnn.deallocate"(%35) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        %38 = "ttnn.to_memory_config"(%34) : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x1x1x1xf32, #ttnn_layout39>
        "ttnn.deallocate"(%34) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        %39 = "ttnn.where"(%36, %38, %37) : (tensor<1x1x1x1xf32, #ttnn_layout39>, tensor<1x1x1x1xf32, #ttnn_layout39>, tensor<1x1x1x1xf32, #ttnn_layout39>) -> tensor<1x1x1x1xf32, #ttnn_layout39>
        "ttnn.deallocate"(%38) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> ()
        "ttnn.deallocate"(%37) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> ()
        "ttnn.deallocate"(%36) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> ()
        %40 = "ttnn.typecast"(%39) : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> tensor<1x1x1x1xsi32, #ttnn_layout37>
        "ttnn.deallocate"(%39) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> ()
        %41 = "ttnn.to_memory_config"(%40) : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> tensor<1x1x1x1xsi32, #ttnn_layout22>
        "ttnn.deallocate"(%40) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> ()
        %42 = "ttnn.to_memory_config"(%41) : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xsi32, #ttnn_layout37>
        "ttnn.deallocate"(%41) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %43 = "ttnn.to_memory_config"(%2) : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> tensor<1x1x1x1xsi32, #ttnn_layout37>
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout22>) -> ()
        %44 = "ttnn.multiply"(%42, %43) : (tensor<1x1x1x1xsi32, #ttnn_layout37>, tensor<1x1x1x1xsi32, #ttnn_layout37>) -> tensor<1x1x1x1xsi32, #ttnn_layout37>
        "ttnn.deallocate"(%42) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> ()
        %45 = "ttnn.reshape"(%44) <{shape = [1 : i32]}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> tensor<1xsi32, #ttnn_layout31>
        "ttnn.deallocate"(%44) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> ()
        %46 = "ttnn.to_memory_config"(%24) : (tensor<1x1x1x512xsi32, #ttnn_layout35>) -> tensor<1x1x1x512xsi32, #ttnn_layout40>
        %47 = "ttnn.add"(%46, %43) : (tensor<1x1x1x512xsi32, #ttnn_layout40>, tensor<1x1x1x1xsi32, #ttnn_layout37>) -> tensor<1x1x1x512xsi32, #ttnn_layout40>
        "ttnn.deallocate"(%43) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> ()
        %48 = "ttnn.gt"(%29, %46) : (tensor<1x1x1x1xsi32, #ttnn_layout37>, tensor<1x1x1x512xsi32, #ttnn_layout40>) -> tensor<1x1x1x512xbf16, #ttnn_layout41>
        "ttnn.deallocate"(%46) <{force = false}> : (tensor<1x1x1x512xsi32, #ttnn_layout40>) -> ()
        "ttnn.deallocate"(%29) <{force = false}> : (tensor<1x1x1x1xsi32, #ttnn_layout37>) -> ()
        %49 = "ttnn.typecast"(%48) : (tensor<1x1x1x512xbf16, #ttnn_layout41>) -> tensor<1x1x1x512xf32, #ttnn_layout42>
        "ttnn.deallocate"(%48) <{force = false}> : (tensor<1x1x1x512xbf16, #ttnn_layout41>) -> ()
        %50 = "ttnn.to_memory_config"(%49) : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> tensor<1x1x1x512xf32, #ttnn_layout43>
        "ttnn.deallocate"(%49) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> ()
        %51 = "ttnn.typecast"(%47) : (tensor<1x1x1x512xsi32, #ttnn_layout40>) -> tensor<1x1x1x512xf32, #ttnn_layout42>
        "ttnn.deallocate"(%47) <{force = false}> : (tensor<1x1x1x512xsi32, #ttnn_layout40>) -> ()
        %52 = "ttnn.to_memory_config"(%51) : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> tensor<1x1x1x512xf32, #ttnn_layout43>
        "ttnn.deallocate"(%51) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> ()
        %53 = "ttnn.typecast"(%24) : (tensor<1x1x1x512xsi32, #ttnn_layout35>) -> tensor<1x1x1x512xf32, #ttnn_layout44>
        "ttnn.deallocate"(%24) <{force = false}> : (tensor<1x1x1x512xsi32, #ttnn_layout35>) -> ()
        %54 = "ttnn.to_memory_config"(%53) : (tensor<1x1x1x512xf32, #ttnn_layout44>) -> tensor<1x1x1x512xf32, #ttnn_layout43>
        "ttnn.deallocate"(%53) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout44>) -> ()
        %55 = "ttnn.to_memory_config"(%50) : (tensor<1x1x1x512xf32, #ttnn_layout43>) -> tensor<1x1x1x512xf32, #ttnn_layout42>
        "ttnn.deallocate"(%50) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout43>) -> ()
        %56 = "ttnn.to_memory_config"(%54) : (tensor<1x1x1x512xf32, #ttnn_layout43>) -> tensor<1x1x1x512xf32, #ttnn_layout42>
        "ttnn.deallocate"(%54) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout43>) -> ()
        %57 = "ttnn.to_memory_config"(%52) : (tensor<1x1x1x512xf32, #ttnn_layout43>) -> tensor<1x1x1x512xf32, #ttnn_layout42>
        "ttnn.deallocate"(%52) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout43>) -> ()
        %58 = "ttnn.where"(%55, %57, %56) : (tensor<1x1x1x512xf32, #ttnn_layout42>, tensor<1x1x1x512xf32, #ttnn_layout42>, tensor<1x1x1x512xf32, #ttnn_layout42>) -> tensor<1x1x1x512xf32, #ttnn_layout42>
        "ttnn.deallocate"(%57) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> ()
        "ttnn.deallocate"(%56) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> ()
        "ttnn.deallocate"(%55) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> ()
        %59 = "ttnn.typecast"(%58) : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> tensor<1x1x1x512xsi32, #ttnn_layout40>
        "ttnn.deallocate"(%58) <{force = false}> : (tensor<1x1x1x512xf32, #ttnn_layout42>) -> ()
        %60 = "ttnn.to_memory_config"(%59) : (tensor<1x1x1x512xsi32, #ttnn_layout40>) -> tensor<1x1x1x512xsi32, #ttnn_layout45>
        "ttnn.deallocate"(%59) <{force = false}> : (tensor<1x1x1x512xsi32, #ttnn_layout40>) -> ()
        %61 = "ttnn.reshape"(%60) <{shape = [512 : i32]}> : (tensor<1x1x1x512xsi32, #ttnn_layout45>) -> tensor<512xsi32, #ttnn_layout29>
        "ttnn.deallocate"(%60) <{force = false}> : (tensor<1x1x1x512xsi32, #ttnn_layout45>) -> ()
        %62 = "ttnn.to_memory_config"(%61) : (tensor<512xsi32, #ttnn_layout29>) -> tensor<512xsi32, #ttnn_layout46>
        "ttnn.deallocate"(%61) <{force = false}> : (tensor<512xsi32, #ttnn_layout29>) -> ()
        %63 = "ttnn.add"(%45, %62) : (tensor<1xsi32, #ttnn_layout31>, tensor<512xsi32, #ttnn_layout46>) -> tensor<512xsi32, #ttnn_layout46>
        "ttnn.deallocate"(%62) <{force = false}> : (tensor<512xsi32, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%45) <{force = false}> : (tensor<1xsi32, #ttnn_layout31>) -> ()
        %64 = "ttnn.full"(%0) <{fill_value = 0 : i32, shape = #ttnn.shape<512>}> : (!ttnn.device) -> tensor<512xsi32, #ttnn_layout29>
        %65 = "ttnn.to_memory_config"(%64) : (tensor<512xsi32, #ttnn_layout29>) -> tensor<512xsi32, #ttnn_layout46>
        "ttnn.deallocate"(%64) <{force = false}> : (tensor<512xsi32, #ttnn_layout29>) -> ()
        %66 = "ttnn.lt"(%63, %65) : (tensor<512xsi32, #ttnn_layout46>, tensor<512xsi32, #ttnn_layout46>) -> tensor<512xbf16, #ttnn_layout47>
        %67 = "ttnn.maximum"(%63, %65) : (tensor<512xsi32, #ttnn_layout46>, tensor<512xsi32, #ttnn_layout46>) -> tensor<512xsi32, #ttnn_layout46>
        "ttnn.deallocate"(%65) <{force = false}> : (tensor<512xsi32, #ttnn_layout46>) -> ()
        "ttnn.deallocate"(%63) <{force = false}> : (tensor<512xsi32, #ttnn_layout46>) -> ()
        %68 = "ttnn.typecast"(%67) : (tensor<512xsi32, #ttnn_layout46>) -> tensor<512xui32, #ttnn_layout48>
        "ttnn.deallocate"(%67) <{force = false}> : (tensor<512xsi32, #ttnn_layout46>) -> ()
        %69 = "ttnn.to_memory_config"(%68) : (tensor<512xui32, #ttnn_layout48>) -> tensor<512xui32, #ttnn_layout49>
        "ttnn.deallocate"(%68) <{force = false}> : (tensor<512xui32, #ttnn_layout48>) -> ()
        %70 = "ttnn.reshape"(%69) <{shape = [1 : i32, 512 : i32]}> : (tensor<512xui32, #ttnn_layout49>) -> tensor<1x512xui32, #ttnn_layout50>
        "ttnn.deallocate"(%69) <{force = false}> : (tensor<512xui32, #ttnn_layout49>) -> ()
        %71 = "ttnn.to_layout"(%70) : (tensor<1x512xui32, #ttnn_layout50>) -> tensor<1x512xui32, #ttnn_layout26>
        "ttnn.deallocate"(%70) <{force = false}> : (tensor<1x512xui32, #ttnn_layout50>) -> ()
        %72 = "ttnn.gather"(%19, %71) <{dim = 1 : si32}> : (tensor<1x512xbf16, #ttnn_layout32>, tensor<1x512xui32, #ttnn_layout26>) -> tensor<1x512xbf16, #ttnn_layout32>
        "ttnn.deallocate"(%71) <{force = false}> : (tensor<1x512xui32, #ttnn_layout26>) -> ()
        "ttnn.deallocate"(%19) <{force = false}> : (tensor<1x512xbf16, #ttnn_layout32>) -> ()
        %73 = "ttnn.to_layout"(%72) : (tensor<1x512xbf16, #ttnn_layout32>) -> tensor<1x512xbf16, #ttnn_layout51>
        "ttnn.deallocate"(%72) <{force = false}> : (tensor<1x512xbf16, #ttnn_layout32>) -> ()
        %74 = "ttnn.reshape"(%73) <{shape = [512 : i32]}> : (tensor<1x512xbf16, #ttnn_layout51>) -> tensor<512xbf16, #ttnn_layout52>
        "ttnn.deallocate"(%73) <{force = false}> : (tensor<1x512xbf16, #ttnn_layout51>) -> ()
        %75 = "ttnn.full"(%0) <{fill_value = 0x7FC00000 : f32, shape = #ttnn.shape<512>}> : (!ttnn.device) -> tensor<512xbf16, #ttnn_layout52>
        %76 = "ttnn.to_memory_config"(%74) : (tensor<512xbf16, #ttnn_layout52>) -> tensor<512xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%74) <{force = false}> : (tensor<512xbf16, #ttnn_layout52>) -> ()
        %77 = "ttnn.to_memory_config"(%75) : (tensor<512xbf16, #ttnn_layout52>) -> tensor<512xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%75) <{force = false}> : (tensor<512xbf16, #ttnn_layout52>) -> ()
        %78 = "ttnn.where"(%66, %77, %76) : (tensor<512xbf16, #ttnn_layout47>, tensor<512xbf16, #ttnn_layout47>, tensor<512xbf16, #ttnn_layout47>) -> tensor<512xbf16, #ttnn_layout47>
        "ttnn.deallocate"(%77) <{force = false}> : (tensor<512xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%76) <{force = false}> : (tensor<512xbf16, #ttnn_layout47>) -> ()
        "ttnn.deallocate"(%66) <{force = false}> : (tensor<512xbf16, #ttnn_layout47>) -> ()
        %79 = "ttnn.to_memory_config"(%78) : (tensor<512xbf16, #ttnn_layout47>) -> tensor<512xbf16, #ttnn_layout53>
        "ttnn.deallocate"(%78) <{force = false}> : (tensor<512xbf16, #ttnn_layout47>) -> ()
        %80 = "ttnn.reshape"(%79) <{shape = [1 : i32, 1 : i32, 1 : i32, 512 : i32]}> : (tensor<512xbf16, #ttnn_layout53>) -> tensor<1x1x1x512xbf16, #ttnn_layout54>
        "ttnn.deallocate"(%79) <{force = false}> : (tensor<512xbf16, #ttnn_layout53>) -> ()
        %81 = "ttnn.logical_and"(%25, %80) : (tensor<1x1x512x512xbf16, #ttnn_layout36>, tensor<1x1x1x512xbf16, #ttnn_layout54>) -> tensor<1x1x512x512xbf16, #ttnn_layout36>
        "ttnn.deallocate"(%80) <{force = false}> : (tensor<1x1x1x512xbf16, #ttnn_layout54>) -> ()
        "ttnn.deallocate"(%25) <{force = false}> : (tensor<1x1x512x512xbf16, #ttnn_layout36>) -> ()
        %82 = "ttnn.reshape"(%arg146) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<32xf32, #ttnn_layout6>) -> tensor<1x32x1xf32, #ttnn_layout10>
        "ttnn.deallocate"(%arg146) <{force = false}> : (tensor<32xf32, #ttnn_layout6>) -> ()
        %83 = "ttnn.typecast"(%22) : (tensor<1x1x512xsi32, #ttnn_layout33>) -> tensor<1x1x512xf32, #ttnn_layout55>
        "ttnn.deallocate"(%22) <{force = false}> : (tensor<1x1x512xsi32, #ttnn_layout33>) -> ()
        %84 = "ttnn.to_memory_config"(%82) : (tensor<1x32x1xf32, #ttnn_layout10>) -> tensor<1x32x1xf32, #ttnn_layout56>
        "ttnn.deallocate"(%82) <{force = false}> : (tensor<1x32x1xf32, #ttnn_layout10>) -> ()
        %85 = "ttnn.to_memory_config"(%83) : (tensor<1x1x512xf32, #ttnn_layout55>) -> tensor<1x1x512xf32, #ttnn_layout57>
        "ttnn.deallocate"(%83) <{force = false}> : (tensor<1x1x512xf32, #ttnn_layout55>) -> ()
        %86 = "ttnn.matmul"(%84, %85) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<1, 1>, in0_block_w = 1, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 1, out_block_w = 16, per_core_m = 1, per_core_n = 16, fuse_batch = true, mcast_in0 = false, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = false}> : (tensor<1x32x1xf32, #ttnn_layout56>, tensor<1x1x512xf32, #ttnn_layout57>) -> tensor<1x32x512xf32, #ttnn_layout58>
        "ttnn.deallocate"(%85) <{force = false}> : (tensor<1x1x512xf32, #ttnn_layout57>) -> ()
        "ttnn.deallocate"(%84) <{force = false}> : (tensor<1x32x1xf32, #ttnn_layout56>) -> ()
        %87 = "ttnn.permute"(%86) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x32x512xf32, #ttnn_layout58>) -> tensor<1x512x32xf32, #ttnn_layout59>
        "ttnn.deallocate"(%86) <{force = false}> : (tensor<1x32x512xf32, #ttnn_layout58>) -> ()
        %88 = "ttnn.to_memory_config"(%87) : (tensor<1x512x32xf32, #ttnn_layout59>) -> tensor<1x512x32xf32, #ttnn_layout60>
        "ttnn.deallocate"(%87) <{force = false}> : (tensor<1x512x32xf32, #ttnn_layout59>) -> ()
        %89 = "ttnn.concat"(%88, %88) <{dim = 2 : si32}> : (tensor<1x512x32xf32, #ttnn_layout60>, tensor<1x512x32xf32, #ttnn_layout60>) -> tensor<1x512x64xf32, #ttnn_layout61>
        "ttnn.deallocate"(%88) <{force = false}> : (tensor<1x512x32xf32, #ttnn_layout60>) -> ()
        %90 = "ttnn.cos"(%89) : (tensor<1x512x64xf32, #ttnn_layout61>) -> tensor<1x512x64xf32, #ttnn_layout61>
        %91 = "ttnn.to_memory_config"(%7) : (tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x1x1xf32, #ttnn_layout56>
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout10>) -> ()
        %92 = "ttnn.multiply"(%90, %91) : (tensor<1x512x64xf32, #ttnn_layout61>, tensor<1x1x1xf32, #ttnn_layout56>) -> tensor<1x512x64xf32, #ttnn_layout61>
        "ttnn.deallocate"(%90) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout61>) -> ()
        %93 = "ttnn.sin"(%89) : (tensor<1x512x64xf32, #ttnn_layout61>) -> tensor<1x512x64xf32, #ttnn_layout61>
        "ttnn.deallocate"(%89) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout61>) -> ()
        %94 = "ttnn.multiply"(%93, %91) : (tensor<1x512x64xf32, #ttnn_layout61>, tensor<1x1x1xf32, #ttnn_layout56>) -> tensor<1x512x64xf32, #ttnn_layout61>
        "ttnn.deallocate"(%93) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout61>) -> ()
        "ttnn.deallocate"(%91) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout56>) -> ()
        %95 = "ttnn.typecast"(%92) : (tensor<1x512x64xf32, #ttnn_layout61>) -> tensor<1x512x64xbf16, #ttnn_layout62>
        "ttnn.deallocate"(%92) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout61>) -> ()
        %96 = "ttnn.typecast"(%94) : (tensor<1x512x64xf32, #ttnn_layout61>) -> tensor<1x512x64xbf16, #ttnn_layout62>
        "ttnn.deallocate"(%94) <{force = false}> : (tensor<1x512x64xf32, #ttnn_layout61>) -> ()
        %97 = "ttnn.to_memory_config"(%14) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        %98 = "ttnn.to_memory_config"(%97) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%97) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %99 = "ttnn.typecast"(%98) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%98) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %100 = "ttnn.to_memory_config"(%99) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %101 = "ttnn.to_memory_config"(%99) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %102 = "ttnn.pow_scalar"(%101) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%101) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %103 = "ttnn.mean"(%102) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%102) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %104 = "ttnn.to_memory_config"(%6) : (tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x1x1xf32, #ttnn_layout56>
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout10>) -> ()
        %105 = "ttnn.to_memory_config"(%104) : (tensor<1x1x1xf32, #ttnn_layout56>) -> tensor<1x1x1xf32, #ttnn_layout10>
        "ttnn.deallocate"(%104) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout56>) -> ()
        %106 = "ttnn.to_memory_config"(%105) : (tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x1x1xf32, #ttnn_layout56>
        %107 = "ttnn.add"(%103, %106) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout56>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%106) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout56>) -> ()
        "ttnn.deallocate"(%103) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %108 = "ttnn.rsqrt"(%107) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%107) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %109 = "ttnn.to_memory_config"(%108) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %110 = "ttnn.multiply"(%99, %108) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%108) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%99) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %111 = "ttnn.typecast"(%110) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%110) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %112 = "ttnn.to_memory_config"(%111) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%111) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %113 = "ttnn.to_memory_config"(%14) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        %114 = "ttnn.to_memory_config"(%113) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%113) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %115 = "ttnn.rms_norm"(%114, %arg8) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%114) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %116 = "ttnn.to_memory_config"(%115) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %117 = "ttnn.concat"(%arg1, %arg2, %arg3) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %118 = "ttnn.matmul"(%115, %117) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%117) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%115) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %119 = "ttnn.to_memory_config"(%118) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%118) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query, %key, %value = "ttnn.split_query_key_value_and_split_heads"(%119) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%119) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %120 = "ttnn.reshape"(%95) <{shape = [1 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x512x64xbf16, #ttnn_layout62>) -> tensor<1x1x512x64xbf16, #ttnn_layout73>
        "ttnn.deallocate"(%95) <{force = false}> : (tensor<1x512x64xbf16, #ttnn_layout62>) -> ()
        %121 = "ttnn.to_memory_config"(%120) : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> tensor<1x1x512x64xbf16, #ttnn_layout16>
        "ttnn.deallocate"(%120) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> ()
        %122 = "ttnn.to_memory_config"(%121) : (tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x1x512x64xbf16, #ttnn_layout73>
        %123 = "ttnn.to_memory_config"(%122) : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> tensor<1x1x512x64xbf16, #ttnn_layout16>
        "ttnn.deallocate"(%122) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> ()
        %124 = "ttnn.reshape"(%96) <{shape = [1 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x512x64xbf16, #ttnn_layout62>) -> tensor<1x1x512x64xbf16, #ttnn_layout73>
        "ttnn.deallocate"(%96) <{force = false}> : (tensor<1x512x64xbf16, #ttnn_layout62>) -> ()
        %125 = "ttnn.to_memory_config"(%124) : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> tensor<1x1x512x64xbf16, #ttnn_layout16>
        "ttnn.deallocate"(%124) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> ()
        %126 = "ttnn.to_memory_config"(%125) : (tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x1x512x64xbf16, #ttnn_layout73>
        %127 = "ttnn.to_memory_config"(%126) : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> tensor<1x1x512x64xbf16, #ttnn_layout16>
        "ttnn.deallocate"(%126) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> ()
        %128 = "ttnn.to_memory_config"(%121) : (tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x1x512x64xbf16, #ttnn_layout73>
        %129 = "ttnn.to_memory_config"(%125) : (tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x1x512x64xbf16, #ttnn_layout73>
        %130 = "ttnn.rotary_embedding"(%query, %128, %129) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout73>, tensor<1x1x512x64xbf16, #ttnn_layout73>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%129) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> ()
        "ttnn.deallocate"(%128) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> ()
        "ttnn.deallocate"(%query) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %131 = "ttnn.to_memory_config"(%121) : (tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x1x512x64xbf16, #ttnn_layout73>
        %132 = "ttnn.to_memory_config"(%125) : (tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x1x512x64xbf16, #ttnn_layout73>
        %133 = "ttnn.rotary_embedding"(%key, %131, %132) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout73>, tensor<1x1x512x64xbf16, #ttnn_layout73>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%132) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> ()
        "ttnn.deallocate"(%131) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout73>) -> ()
        "ttnn.deallocate"(%key) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %134 = "ttnn.reshape"(%133) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%133) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %135 = "ttnn.reshape"(%value) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %136 = "ttnn.to_memory_config"(%4) : (tensor<1x1x1x1xbf16, #ttnn_layout24>) -> tensor<1x1x1x1xbf16, #ttnn_layout38>
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout24>) -> ()
        %137 = "ttnn.to_memory_config"(%5) : (tensor<1x1x1x1xbf16, #ttnn_layout24>) -> tensor<1x1x1x1xbf16, #ttnn_layout38>
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout24>) -> ()
        %138 = "ttnn.where"(%81, %137, %136) : (tensor<1x1x512x512xbf16, #ttnn_layout36>, tensor<1x1x1x1xbf16, #ttnn_layout38>, tensor<1x1x1x1xbf16, #ttnn_layout38>) -> tensor<1x1x512x512xbf16, #ttnn_layout36>
        "ttnn.deallocate"(%137) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout38>) -> ()
        "ttnn.deallocate"(%136) <{force = false}> : (tensor<1x1x1x1xbf16, #ttnn_layout38>) -> ()
        "ttnn.deallocate"(%81) <{force = false}> : (tensor<1x1x512x512xbf16, #ttnn_layout36>) -> ()
        %139 = "ttnn.to_memory_config"(%130) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%130) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %140 = "ttnn.typecast"(%139) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%139) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %141 = "ttnn.to_memory_config"(%134) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%134) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %142 = "ttnn.typecast"(%141) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%141) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %143 = "ttnn.to_memory_config"(%135) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%135) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %144 = "ttnn.typecast"(%143) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%143) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %145 = "ttnn.repeat"(%144) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%144) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %146 = "ttnn.reshape"(%145) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%145) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %147 = "ttnn.to_memory_config"(%146) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%146) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %148 = "ttnn.to_memory_config"(%3) : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x1x1x1xf32, #ttnn_layout39>
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        %149 = "ttnn.to_memory_config"(%148) : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> tensor<1x1x1x1xf32, #ttnn_layout23>
        "ttnn.deallocate"(%148) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> ()
        %150 = "ttnn.to_memory_config"(%149) : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x1x1x1xf32, #ttnn_layout39>
        %151 = "ttnn.multiply"(%140, %150) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout39>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%150) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout39>) -> ()
        "ttnn.deallocate"(%140) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %152 = "ttnn.to_memory_config"(%151) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %153 = "ttnn.to_memory_config"(%1) : (tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x1x1x1x1xf32, #ttnn_layout80>
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> ()
        %154 = "ttnn.to_memory_config"(%153) : (tensor<1x1x1x1x1xf32, #ttnn_layout80>) -> tensor<1x1x1x1x1xf32, #ttnn_layout21>
        "ttnn.deallocate"(%153) <{force = false}> : (tensor<1x1x1x1x1xf32, #ttnn_layout80>) -> ()
        %155 = "ttnn.to_memory_config"(%154) : (tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x1x1x1x1xf32, #ttnn_layout80>
        %156 = "ttnn.multiply"(%142, %155) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout80>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%155) <{force = false}> : (tensor<1x1x1x1x1xf32, #ttnn_layout80>) -> ()
        "ttnn.deallocate"(%142) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %157 = "ttnn.repeat"(%156) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%156) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %158 = "ttnn.reshape"(%157) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%157) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %159 = "ttnn.permute"(%158) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %160 = "ttnn.to_memory_config"(%159) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%159) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %161 = "ttnn.typecast"(%138) : (tensor<1x1x512x512xbf16, #ttnn_layout36>) -> tensor<1x1x512x512xf32, #ttnn_layout82>
        "ttnn.deallocate"(%138) <{force = false}> : (tensor<1x1x512x512xbf16, #ttnn_layout36>) -> ()
        %162 = "ttnn.to_memory_config"(%161) : (tensor<1x1x512x512xf32, #ttnn_layout82>) -> tensor<1x1x512x512xf32, #ttnn_layout83>
        "ttnn.deallocate"(%161) <{force = false}> : (tensor<1x1x512x512xf32, #ttnn_layout82>) -> ()
        %163 = "ttnn.to_memory_config"(%158) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%158) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %164 = "ttnn.matmul"(%151, %163) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%163) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%151) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %165 = "ttnn.to_memory_config"(%164) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%164) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %166 = "ttnn.to_memory_config"(%162) : (tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x1x512x512xf32, #ttnn_layout82>
        %167 = "ttnn.add"(%165, %166) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout82>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%166) <{force = false}> : (tensor<1x1x512x512xf32, #ttnn_layout82>) -> ()
        "ttnn.deallocate"(%165) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %168 = "ttnn.to_memory_config"(%167) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%167) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %169 = "ttnn.softmax"(%168) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%168) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %170 = "ttnn.to_memory_config"(%169) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %171 = "ttnn.matmul"(%169, %147) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%169) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %172 = "ttnn.to_memory_config"(%171) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%171) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %173 = "ttnn.typecast"(%172) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%172) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %174 = "ttnn.to_memory_config"(%173) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%173) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %175 = "ttnn.concatenate_heads"(%174) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%174) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %176 = "ttnn.to_memory_config"(%175) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %177 = "ttnn.matmul"(%175, %arg4) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%175) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %178 = "ttnn.add"(%177, %14) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%177) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%14) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %179 = "ttnn.typecast"(%178) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %180 = "ttnn.to_memory_config"(%179) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %181 = "ttnn.to_memory_config"(%179) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %182 = "ttnn.pow_scalar"(%181) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%181) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %183 = "ttnn.mean"(%182) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%182) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %184 = "ttnn.add"(%183, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%183) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %185 = "ttnn.rsqrt"(%184) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%184) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %186 = "ttnn.to_memory_config"(%185) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %187 = "ttnn.multiply"(%179, %185) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%185) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%179) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %188 = "ttnn.typecast"(%187) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%187) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %189 = "ttnn.to_memory_config"(%188) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%188) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %190 = "ttnn.to_memory_config"(%178) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %191 = "ttnn.rms_norm"(%190, %arg9) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%190) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %192 = "ttnn.to_memory_config"(%191) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %193 = "ttnn.matmul"(%191, %arg5) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %194 = "ttnn.to_memory_config"(%193) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %195 = "ttnn.silu"(%193) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%193) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %196 = "ttnn.to_memory_config"(%195) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %197 = "ttnn.matmul"(%191, %arg6) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%191) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %198 = "ttnn.to_memory_config"(%197) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %199 = "ttnn.multiply"(%195, %197) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%197) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%195) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %200 = "ttnn.to_memory_config"(%199) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %201 = "ttnn.matmul"(%199, %arg7) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%199) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %202 = "ttnn.add"(%201, %178) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%201) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%178) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %203 = "ttnn.to_memory_config"(%202) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%202) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %204 = "ttnn.to_memory_config"(%203) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %205 = "ttnn.typecast"(%204) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%204) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %206 = "ttnn.to_memory_config"(%205) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %207 = "ttnn.pow_scalar"(%205) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %208 = "ttnn.mean"(%207) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%207) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %209 = "ttnn.add"(%208, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%208) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %210 = "ttnn.rsqrt"(%209) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%209) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %211 = "ttnn.to_memory_config"(%210) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %212 = "ttnn.multiply"(%205, %210) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%210) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%205) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %213 = "ttnn.typecast"(%212) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%212) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %214 = "ttnn.to_memory_config"(%213) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%213) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %215 = "ttnn.to_memory_config"(%203) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %216 = "ttnn.rms_norm"(%215, %arg17) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%215) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %217 = "ttnn.to_memory_config"(%216) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %218 = "ttnn.concat"(%arg10, %arg11, %arg12) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %219 = "ttnn.matmul"(%216, %218) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%218) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%216) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %220 = "ttnn.to_memory_config"(%219) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%219) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_0, %key_1, %value_2 = "ttnn.split_query_key_value_and_split_heads"(%220) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%220) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %221 = "ttnn.rotary_embedding"(%query_0, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_0) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %222 = "ttnn.rotary_embedding"(%key_1, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_1) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %223 = "ttnn.reshape"(%222) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%222) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %224 = "ttnn.reshape"(%value_2) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_2) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %225 = "ttnn.to_memory_config"(%221) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%221) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %226 = "ttnn.typecast"(%225) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%225) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %227 = "ttnn.to_memory_config"(%223) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%223) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %228 = "ttnn.typecast"(%227) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%227) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %229 = "ttnn.to_memory_config"(%224) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%224) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %230 = "ttnn.typecast"(%229) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%229) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %231 = "ttnn.repeat"(%230) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%230) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %232 = "ttnn.reshape"(%231) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%231) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %233 = "ttnn.to_memory_config"(%232) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%232) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %234 = "ttnn.multiply"(%226, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%226) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %235 = "ttnn.to_memory_config"(%234) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %236 = "ttnn.multiply"(%228, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%228) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %237 = "ttnn.repeat"(%236) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%236) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %238 = "ttnn.reshape"(%237) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%237) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %239 = "ttnn.permute"(%238) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %240 = "ttnn.to_memory_config"(%239) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%239) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %241 = "ttnn.to_memory_config"(%238) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%238) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %242 = "ttnn.matmul"(%234, %241) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%241) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%234) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %243 = "ttnn.to_memory_config"(%242) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%242) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %244 = "ttnn.add"(%243, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%243) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %245 = "ttnn.to_memory_config"(%244) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%244) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %246 = "ttnn.softmax"(%245) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%245) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %247 = "ttnn.to_memory_config"(%246) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %248 = "ttnn.matmul"(%246, %233) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%246) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %249 = "ttnn.to_memory_config"(%248) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%248) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %250 = "ttnn.typecast"(%249) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%249) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %251 = "ttnn.to_memory_config"(%250) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%250) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %252 = "ttnn.concatenate_heads"(%251) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%251) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %253 = "ttnn.to_memory_config"(%252) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %254 = "ttnn.matmul"(%252, %arg13) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%252) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %255 = "ttnn.add"(%254, %203) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%254) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%203) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %256 = "ttnn.typecast"(%255) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %257 = "ttnn.to_memory_config"(%256) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %258 = "ttnn.pow_scalar"(%256) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %259 = "ttnn.mean"(%258) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%258) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %260 = "ttnn.add"(%259, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%259) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %261 = "ttnn.rsqrt"(%260) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%260) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %262 = "ttnn.to_memory_config"(%261) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %263 = "ttnn.multiply"(%256, %261) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%261) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%256) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %264 = "ttnn.typecast"(%263) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%263) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %265 = "ttnn.to_memory_config"(%264) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%264) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %266 = "ttnn.rms_norm"(%255, %arg18) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %267 = "ttnn.to_memory_config"(%266) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %268 = "ttnn.to_memory_config"(%266) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%266) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %269 = "ttnn.matmul"(%268, %arg14) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %270 = "ttnn.to_memory_config"(%269) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %271 = "ttnn.silu"(%269) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%269) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %272 = "ttnn.to_memory_config"(%271) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %273 = "ttnn.matmul"(%268, %arg15) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%268) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %274 = "ttnn.to_memory_config"(%273) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %275 = "ttnn.multiply"(%271, %273) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%273) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%271) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %276 = "ttnn.to_memory_config"(%275) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %277 = "ttnn.matmul"(%275, %arg16) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%275) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %278 = "ttnn.add"(%277, %255) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%277) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%255) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %279 = "ttnn.to_memory_config"(%278) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%278) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %280 = "ttnn.to_memory_config"(%279) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %281 = "ttnn.typecast"(%280) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%280) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %282 = "ttnn.to_memory_config"(%281) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %283 = "ttnn.pow_scalar"(%281) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %284 = "ttnn.mean"(%283) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%283) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %285 = "ttnn.add"(%284, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%284) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %286 = "ttnn.rsqrt"(%285) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%285) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %287 = "ttnn.to_memory_config"(%286) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %288 = "ttnn.multiply"(%281, %286) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%286) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%281) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %289 = "ttnn.typecast"(%288) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%288) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %290 = "ttnn.to_memory_config"(%289) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%289) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %291 = "ttnn.to_memory_config"(%279) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %292 = "ttnn.rms_norm"(%291, %arg26) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%291) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %293 = "ttnn.to_memory_config"(%292) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %294 = "ttnn.concat"(%arg19, %arg20, %arg21) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %295 = "ttnn.matmul"(%292, %294) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%294) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%292) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %296 = "ttnn.to_memory_config"(%295) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%295) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_3, %key_4, %value_5 = "ttnn.split_query_key_value_and_split_heads"(%296) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%296) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %297 = "ttnn.rotary_embedding"(%query_3, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_3) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %298 = "ttnn.rotary_embedding"(%key_4, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_4) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %299 = "ttnn.reshape"(%298) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%298) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %300 = "ttnn.reshape"(%value_5) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_5) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %301 = "ttnn.to_memory_config"(%297) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%297) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %302 = "ttnn.typecast"(%301) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%301) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %303 = "ttnn.to_memory_config"(%299) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%299) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %304 = "ttnn.typecast"(%303) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%303) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %305 = "ttnn.to_memory_config"(%300) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%300) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %306 = "ttnn.typecast"(%305) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%305) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %307 = "ttnn.repeat"(%306) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%306) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %308 = "ttnn.reshape"(%307) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%307) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %309 = "ttnn.to_memory_config"(%308) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%308) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %310 = "ttnn.multiply"(%302, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%302) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %311 = "ttnn.to_memory_config"(%310) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %312 = "ttnn.multiply"(%304, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%304) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %313 = "ttnn.repeat"(%312) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%312) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %314 = "ttnn.reshape"(%313) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%313) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %315 = "ttnn.permute"(%314) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %316 = "ttnn.to_memory_config"(%315) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%315) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %317 = "ttnn.to_memory_config"(%314) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%314) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %318 = "ttnn.matmul"(%310, %317) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%317) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%310) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %319 = "ttnn.to_memory_config"(%318) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%318) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %320 = "ttnn.add"(%319, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%319) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %321 = "ttnn.to_memory_config"(%320) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%320) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %322 = "ttnn.softmax"(%321) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%321) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %323 = "ttnn.to_memory_config"(%322) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %324 = "ttnn.matmul"(%322, %309) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%322) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %325 = "ttnn.to_memory_config"(%324) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%324) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %326 = "ttnn.typecast"(%325) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%325) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %327 = "ttnn.to_memory_config"(%326) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%326) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %328 = "ttnn.concatenate_heads"(%327) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%327) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %329 = "ttnn.to_memory_config"(%328) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %330 = "ttnn.matmul"(%328, %arg22) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%328) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %331 = "ttnn.add"(%330, %279) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%330) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%279) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %332 = "ttnn.typecast"(%331) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %333 = "ttnn.to_memory_config"(%332) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %334 = "ttnn.pow_scalar"(%332) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %335 = "ttnn.mean"(%334) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%334) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %336 = "ttnn.add"(%335, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%335) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %337 = "ttnn.rsqrt"(%336) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%336) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %338 = "ttnn.to_memory_config"(%337) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %339 = "ttnn.multiply"(%332, %337) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%337) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%332) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %340 = "ttnn.typecast"(%339) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%339) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %341 = "ttnn.to_memory_config"(%340) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%340) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %342 = "ttnn.rms_norm"(%331, %arg27) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %343 = "ttnn.to_memory_config"(%342) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %344 = "ttnn.to_memory_config"(%342) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%342) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %345 = "ttnn.matmul"(%344, %arg23) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %346 = "ttnn.to_memory_config"(%345) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %347 = "ttnn.silu"(%345) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%345) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %348 = "ttnn.to_memory_config"(%347) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %349 = "ttnn.matmul"(%344, %arg24) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%344) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %350 = "ttnn.to_memory_config"(%349) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %351 = "ttnn.multiply"(%347, %349) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%349) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%347) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %352 = "ttnn.to_memory_config"(%351) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %353 = "ttnn.matmul"(%351, %arg25) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%351) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %354 = "ttnn.add"(%353, %331) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%353) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%331) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %355 = "ttnn.to_memory_config"(%354) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%354) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %356 = "ttnn.to_memory_config"(%355) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %357 = "ttnn.typecast"(%356) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%356) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %358 = "ttnn.to_memory_config"(%357) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %359 = "ttnn.pow_scalar"(%357) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %360 = "ttnn.mean"(%359) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%359) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %361 = "ttnn.add"(%360, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%360) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %362 = "ttnn.rsqrt"(%361) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%361) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %363 = "ttnn.to_memory_config"(%362) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %364 = "ttnn.multiply"(%357, %362) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%362) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%357) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %365 = "ttnn.typecast"(%364) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%364) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %366 = "ttnn.to_memory_config"(%365) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%365) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %367 = "ttnn.to_memory_config"(%355) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %368 = "ttnn.rms_norm"(%367, %arg35) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%367) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %369 = "ttnn.to_memory_config"(%368) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %370 = "ttnn.concat"(%arg28, %arg29, %arg30) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %371 = "ttnn.matmul"(%368, %370) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%370) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%368) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %372 = "ttnn.to_memory_config"(%371) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%371) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_6, %key_7, %value_8 = "ttnn.split_query_key_value_and_split_heads"(%372) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%372) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %373 = "ttnn.rotary_embedding"(%query_6, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_6) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %374 = "ttnn.rotary_embedding"(%key_7, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_7) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %375 = "ttnn.reshape"(%374) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%374) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %376 = "ttnn.reshape"(%value_8) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_8) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %377 = "ttnn.to_memory_config"(%373) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%373) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %378 = "ttnn.typecast"(%377) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%377) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %379 = "ttnn.to_memory_config"(%375) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%375) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %380 = "ttnn.typecast"(%379) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%379) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %381 = "ttnn.to_memory_config"(%376) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%376) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %382 = "ttnn.typecast"(%381) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%381) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %383 = "ttnn.repeat"(%382) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%382) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %384 = "ttnn.reshape"(%383) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%383) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %385 = "ttnn.to_memory_config"(%384) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%384) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %386 = "ttnn.multiply"(%378, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%378) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %387 = "ttnn.to_memory_config"(%386) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %388 = "ttnn.multiply"(%380, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%380) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %389 = "ttnn.repeat"(%388) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%388) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %390 = "ttnn.reshape"(%389) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%389) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %391 = "ttnn.permute"(%390) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %392 = "ttnn.to_memory_config"(%391) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%391) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %393 = "ttnn.to_memory_config"(%390) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%390) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %394 = "ttnn.matmul"(%386, %393) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%393) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%386) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %395 = "ttnn.to_memory_config"(%394) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%394) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %396 = "ttnn.add"(%395, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%395) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %397 = "ttnn.to_memory_config"(%396) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%396) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %398 = "ttnn.softmax"(%397) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%397) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %399 = "ttnn.to_memory_config"(%398) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %400 = "ttnn.matmul"(%398, %385) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%398) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %401 = "ttnn.to_memory_config"(%400) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%400) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %402 = "ttnn.typecast"(%401) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%401) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %403 = "ttnn.to_memory_config"(%402) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%402) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %404 = "ttnn.concatenate_heads"(%403) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%403) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %405 = "ttnn.to_memory_config"(%404) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %406 = "ttnn.matmul"(%404, %arg31) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%404) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %407 = "ttnn.add"(%406, %355) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%406) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%355) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %408 = "ttnn.typecast"(%407) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %409 = "ttnn.to_memory_config"(%408) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %410 = "ttnn.pow_scalar"(%408) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %411 = "ttnn.mean"(%410) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%410) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %412 = "ttnn.add"(%411, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%411) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %413 = "ttnn.rsqrt"(%412) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%412) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %414 = "ttnn.to_memory_config"(%413) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %415 = "ttnn.multiply"(%408, %413) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%413) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%408) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %416 = "ttnn.typecast"(%415) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%415) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %417 = "ttnn.to_memory_config"(%416) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%416) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %418 = "ttnn.rms_norm"(%407, %arg36) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %419 = "ttnn.to_memory_config"(%418) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %420 = "ttnn.to_memory_config"(%418) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%418) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %421 = "ttnn.matmul"(%420, %arg32) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %422 = "ttnn.to_memory_config"(%421) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %423 = "ttnn.silu"(%421) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%421) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %424 = "ttnn.to_memory_config"(%423) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %425 = "ttnn.matmul"(%420, %arg33) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%420) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %426 = "ttnn.to_memory_config"(%425) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %427 = "ttnn.multiply"(%423, %425) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%425) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%423) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %428 = "ttnn.to_memory_config"(%427) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %429 = "ttnn.matmul"(%427, %arg34) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%427) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %430 = "ttnn.add"(%429, %407) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%429) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%407) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %431 = "ttnn.to_memory_config"(%430) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%430) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %432 = "ttnn.to_memory_config"(%431) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %433 = "ttnn.typecast"(%432) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%432) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %434 = "ttnn.to_memory_config"(%433) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %435 = "ttnn.pow_scalar"(%433) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %436 = "ttnn.mean"(%435) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%435) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %437 = "ttnn.add"(%436, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%436) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %438 = "ttnn.rsqrt"(%437) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%437) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %439 = "ttnn.to_memory_config"(%438) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %440 = "ttnn.multiply"(%433, %438) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%438) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%433) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %441 = "ttnn.typecast"(%440) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%440) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %442 = "ttnn.to_memory_config"(%441) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%441) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %443 = "ttnn.to_memory_config"(%431) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %444 = "ttnn.rms_norm"(%443, %arg44) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%443) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %445 = "ttnn.to_memory_config"(%444) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %446 = "ttnn.concat"(%arg37, %arg38, %arg39) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %447 = "ttnn.matmul"(%444, %446) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%446) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%444) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %448 = "ttnn.to_memory_config"(%447) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%447) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_9, %key_10, %value_11 = "ttnn.split_query_key_value_and_split_heads"(%448) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%448) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %449 = "ttnn.rotary_embedding"(%query_9, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_9) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %450 = "ttnn.rotary_embedding"(%key_10, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_10) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %451 = "ttnn.reshape"(%450) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%450) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %452 = "ttnn.reshape"(%value_11) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_11) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %453 = "ttnn.to_memory_config"(%449) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%449) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %454 = "ttnn.typecast"(%453) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%453) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %455 = "ttnn.to_memory_config"(%451) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%451) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %456 = "ttnn.typecast"(%455) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%455) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %457 = "ttnn.to_memory_config"(%452) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%452) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %458 = "ttnn.typecast"(%457) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%457) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %459 = "ttnn.repeat"(%458) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%458) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %460 = "ttnn.reshape"(%459) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%459) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %461 = "ttnn.to_memory_config"(%460) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%460) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %462 = "ttnn.multiply"(%454, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%454) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %463 = "ttnn.to_memory_config"(%462) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %464 = "ttnn.multiply"(%456, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%456) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %465 = "ttnn.repeat"(%464) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%464) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %466 = "ttnn.reshape"(%465) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%465) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %467 = "ttnn.permute"(%466) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %468 = "ttnn.to_memory_config"(%467) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%467) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %469 = "ttnn.to_memory_config"(%466) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%466) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %470 = "ttnn.matmul"(%462, %469) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%469) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%462) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %471 = "ttnn.to_memory_config"(%470) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%470) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %472 = "ttnn.add"(%471, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%471) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %473 = "ttnn.to_memory_config"(%472) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%472) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %474 = "ttnn.softmax"(%473) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%473) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %475 = "ttnn.to_memory_config"(%474) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %476 = "ttnn.matmul"(%474, %461) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%474) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %477 = "ttnn.to_memory_config"(%476) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%476) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %478 = "ttnn.typecast"(%477) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%477) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %479 = "ttnn.to_memory_config"(%478) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%478) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %480 = "ttnn.concatenate_heads"(%479) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%479) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %481 = "ttnn.to_memory_config"(%480) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %482 = "ttnn.matmul"(%480, %arg40) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%480) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %483 = "ttnn.add"(%482, %431) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%482) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%431) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %484 = "ttnn.typecast"(%483) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %485 = "ttnn.to_memory_config"(%484) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %486 = "ttnn.pow_scalar"(%484) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %487 = "ttnn.mean"(%486) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%486) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %488 = "ttnn.add"(%487, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%487) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %489 = "ttnn.rsqrt"(%488) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%488) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %490 = "ttnn.to_memory_config"(%489) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %491 = "ttnn.multiply"(%484, %489) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%489) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%484) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %492 = "ttnn.typecast"(%491) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%491) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %493 = "ttnn.to_memory_config"(%492) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%492) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %494 = "ttnn.rms_norm"(%483, %arg45) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %495 = "ttnn.to_memory_config"(%494) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %496 = "ttnn.to_memory_config"(%494) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%494) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %497 = "ttnn.matmul"(%496, %arg41) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %498 = "ttnn.to_memory_config"(%497) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %499 = "ttnn.silu"(%497) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%497) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %500 = "ttnn.to_memory_config"(%499) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %501 = "ttnn.matmul"(%496, %arg42) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%496) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %502 = "ttnn.to_memory_config"(%501) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %503 = "ttnn.multiply"(%499, %501) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%501) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%499) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %504 = "ttnn.to_memory_config"(%503) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %505 = "ttnn.matmul"(%503, %arg43) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%503) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %506 = "ttnn.add"(%505, %483) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%505) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%483) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %507 = "ttnn.to_memory_config"(%506) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%506) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %508 = "ttnn.to_memory_config"(%507) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %509 = "ttnn.typecast"(%508) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%508) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %510 = "ttnn.to_memory_config"(%509) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %511 = "ttnn.pow_scalar"(%509) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %512 = "ttnn.mean"(%511) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%511) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %513 = "ttnn.add"(%512, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%512) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %514 = "ttnn.rsqrt"(%513) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%513) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %515 = "ttnn.to_memory_config"(%514) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %516 = "ttnn.multiply"(%509, %514) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%514) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%509) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %517 = "ttnn.typecast"(%516) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%516) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %518 = "ttnn.to_memory_config"(%517) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%517) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %519 = "ttnn.to_memory_config"(%507) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %520 = "ttnn.rms_norm"(%519, %arg53) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%519) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %521 = "ttnn.to_memory_config"(%520) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %522 = "ttnn.concat"(%arg46, %arg47, %arg48) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %523 = "ttnn.matmul"(%520, %522) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%522) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%520) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %524 = "ttnn.to_memory_config"(%523) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%523) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_12, %key_13, %value_14 = "ttnn.split_query_key_value_and_split_heads"(%524) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%524) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %525 = "ttnn.rotary_embedding"(%query_12, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_12) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %526 = "ttnn.rotary_embedding"(%key_13, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_13) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %527 = "ttnn.reshape"(%526) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%526) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %528 = "ttnn.reshape"(%value_14) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_14) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %529 = "ttnn.to_memory_config"(%525) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%525) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %530 = "ttnn.typecast"(%529) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%529) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %531 = "ttnn.to_memory_config"(%527) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%527) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %532 = "ttnn.typecast"(%531) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%531) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %533 = "ttnn.to_memory_config"(%528) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%528) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %534 = "ttnn.typecast"(%533) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%533) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %535 = "ttnn.repeat"(%534) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%534) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %536 = "ttnn.reshape"(%535) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%535) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %537 = "ttnn.to_memory_config"(%536) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%536) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %538 = "ttnn.multiply"(%530, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%530) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %539 = "ttnn.to_memory_config"(%538) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %540 = "ttnn.multiply"(%532, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%532) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %541 = "ttnn.repeat"(%540) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%540) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %542 = "ttnn.reshape"(%541) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%541) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %543 = "ttnn.permute"(%542) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %544 = "ttnn.to_memory_config"(%543) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%543) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %545 = "ttnn.to_memory_config"(%542) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%542) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %546 = "ttnn.matmul"(%538, %545) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%545) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%538) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %547 = "ttnn.to_memory_config"(%546) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%546) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %548 = "ttnn.add"(%547, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%547) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %549 = "ttnn.to_memory_config"(%548) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%548) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %550 = "ttnn.softmax"(%549) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%549) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %551 = "ttnn.to_memory_config"(%550) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %552 = "ttnn.matmul"(%550, %537) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%550) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %553 = "ttnn.to_memory_config"(%552) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%552) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %554 = "ttnn.typecast"(%553) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%553) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %555 = "ttnn.to_memory_config"(%554) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%554) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %556 = "ttnn.concatenate_heads"(%555) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%555) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %557 = "ttnn.to_memory_config"(%556) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %558 = "ttnn.matmul"(%556, %arg49) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%556) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %559 = "ttnn.add"(%558, %507) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%558) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%507) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %560 = "ttnn.typecast"(%559) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %561 = "ttnn.to_memory_config"(%560) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %562 = "ttnn.pow_scalar"(%560) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %563 = "ttnn.mean"(%562) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%562) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %564 = "ttnn.add"(%563, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%563) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %565 = "ttnn.rsqrt"(%564) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%564) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %566 = "ttnn.to_memory_config"(%565) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %567 = "ttnn.multiply"(%560, %565) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%565) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%560) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %568 = "ttnn.typecast"(%567) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%567) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %569 = "ttnn.to_memory_config"(%568) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%568) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %570 = "ttnn.rms_norm"(%559, %arg54) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %571 = "ttnn.to_memory_config"(%570) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %572 = "ttnn.to_memory_config"(%570) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%570) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %573 = "ttnn.matmul"(%572, %arg50) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %574 = "ttnn.to_memory_config"(%573) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %575 = "ttnn.silu"(%573) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%573) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %576 = "ttnn.to_memory_config"(%575) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %577 = "ttnn.matmul"(%572, %arg51) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%572) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %578 = "ttnn.to_memory_config"(%577) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %579 = "ttnn.multiply"(%575, %577) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%577) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%575) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %580 = "ttnn.to_memory_config"(%579) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %581 = "ttnn.matmul"(%579, %arg52) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%579) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %582 = "ttnn.add"(%581, %559) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%581) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%559) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %583 = "ttnn.to_memory_config"(%582) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%582) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %584 = "ttnn.to_memory_config"(%583) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %585 = "ttnn.typecast"(%584) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%584) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %586 = "ttnn.to_memory_config"(%585) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %587 = "ttnn.pow_scalar"(%585) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %588 = "ttnn.mean"(%587) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%587) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %589 = "ttnn.add"(%588, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%588) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %590 = "ttnn.rsqrt"(%589) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%589) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %591 = "ttnn.to_memory_config"(%590) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %592 = "ttnn.multiply"(%585, %590) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%590) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%585) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %593 = "ttnn.typecast"(%592) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%592) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %594 = "ttnn.to_memory_config"(%593) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%593) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %595 = "ttnn.to_memory_config"(%583) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %596 = "ttnn.rms_norm"(%595, %arg62) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%595) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %597 = "ttnn.to_memory_config"(%596) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %598 = "ttnn.concat"(%arg55, %arg56, %arg57) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %599 = "ttnn.matmul"(%596, %598) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%598) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%596) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %600 = "ttnn.to_memory_config"(%599) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%599) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_15, %key_16, %value_17 = "ttnn.split_query_key_value_and_split_heads"(%600) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%600) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %601 = "ttnn.rotary_embedding"(%query_15, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_15) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %602 = "ttnn.rotary_embedding"(%key_16, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_16) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %603 = "ttnn.reshape"(%602) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%602) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %604 = "ttnn.reshape"(%value_17) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_17) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %605 = "ttnn.to_memory_config"(%601) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%601) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %606 = "ttnn.typecast"(%605) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%605) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %607 = "ttnn.to_memory_config"(%603) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%603) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %608 = "ttnn.typecast"(%607) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%607) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %609 = "ttnn.to_memory_config"(%604) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%604) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %610 = "ttnn.typecast"(%609) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%609) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %611 = "ttnn.repeat"(%610) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%610) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %612 = "ttnn.reshape"(%611) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%611) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %613 = "ttnn.to_memory_config"(%612) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%612) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %614 = "ttnn.multiply"(%606, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%606) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %615 = "ttnn.to_memory_config"(%614) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %616 = "ttnn.multiply"(%608, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%608) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %617 = "ttnn.repeat"(%616) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%616) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %618 = "ttnn.reshape"(%617) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%617) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %619 = "ttnn.permute"(%618) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %620 = "ttnn.to_memory_config"(%619) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%619) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %621 = "ttnn.to_memory_config"(%618) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%618) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %622 = "ttnn.matmul"(%614, %621) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%621) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%614) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %623 = "ttnn.to_memory_config"(%622) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%622) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %624 = "ttnn.add"(%623, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%623) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %625 = "ttnn.to_memory_config"(%624) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%624) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %626 = "ttnn.softmax"(%625) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%625) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %627 = "ttnn.to_memory_config"(%626) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %628 = "ttnn.matmul"(%626, %613) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%626) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %629 = "ttnn.to_memory_config"(%628) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%628) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %630 = "ttnn.typecast"(%629) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%629) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %631 = "ttnn.to_memory_config"(%630) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%630) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %632 = "ttnn.concatenate_heads"(%631) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%631) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %633 = "ttnn.to_memory_config"(%632) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %634 = "ttnn.matmul"(%632, %arg58) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%632) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %635 = "ttnn.add"(%634, %583) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%634) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%583) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %636 = "ttnn.typecast"(%635) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %637 = "ttnn.to_memory_config"(%636) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %638 = "ttnn.pow_scalar"(%636) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %639 = "ttnn.mean"(%638) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%638) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %640 = "ttnn.add"(%639, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%639) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %641 = "ttnn.rsqrt"(%640) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%640) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %642 = "ttnn.to_memory_config"(%641) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %643 = "ttnn.multiply"(%636, %641) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%641) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%636) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %644 = "ttnn.typecast"(%643) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%643) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %645 = "ttnn.to_memory_config"(%644) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%644) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %646 = "ttnn.rms_norm"(%635, %arg63) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %647 = "ttnn.to_memory_config"(%646) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %648 = "ttnn.to_memory_config"(%646) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%646) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %649 = "ttnn.matmul"(%648, %arg59) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %650 = "ttnn.to_memory_config"(%649) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %651 = "ttnn.silu"(%649) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%649) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %652 = "ttnn.to_memory_config"(%651) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %653 = "ttnn.matmul"(%648, %arg60) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%648) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %654 = "ttnn.to_memory_config"(%653) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %655 = "ttnn.multiply"(%651, %653) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%653) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%651) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %656 = "ttnn.to_memory_config"(%655) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %657 = "ttnn.matmul"(%655, %arg61) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%655) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %658 = "ttnn.add"(%657, %635) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%657) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%635) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %659 = "ttnn.to_memory_config"(%658) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%658) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %660 = "ttnn.to_memory_config"(%659) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %661 = "ttnn.typecast"(%660) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%660) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %662 = "ttnn.to_memory_config"(%661) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %663 = "ttnn.pow_scalar"(%661) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %664 = "ttnn.mean"(%663) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%663) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %665 = "ttnn.add"(%664, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%664) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %666 = "ttnn.rsqrt"(%665) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%665) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %667 = "ttnn.to_memory_config"(%666) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %668 = "ttnn.multiply"(%661, %666) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%666) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%661) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %669 = "ttnn.typecast"(%668) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%668) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %670 = "ttnn.to_memory_config"(%669) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%669) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %671 = "ttnn.to_memory_config"(%659) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %672 = "ttnn.rms_norm"(%671, %arg71) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%671) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %673 = "ttnn.to_memory_config"(%672) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %674 = "ttnn.concat"(%arg64, %arg65, %arg66) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %675 = "ttnn.matmul"(%672, %674) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%674) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%672) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %676 = "ttnn.to_memory_config"(%675) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%675) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_18, %key_19, %value_20 = "ttnn.split_query_key_value_and_split_heads"(%676) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%676) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %677 = "ttnn.rotary_embedding"(%query_18, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_18) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %678 = "ttnn.rotary_embedding"(%key_19, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_19) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %679 = "ttnn.reshape"(%678) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%678) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %680 = "ttnn.reshape"(%value_20) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_20) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %681 = "ttnn.to_memory_config"(%677) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%677) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %682 = "ttnn.typecast"(%681) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%681) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %683 = "ttnn.to_memory_config"(%679) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%679) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %684 = "ttnn.typecast"(%683) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%683) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %685 = "ttnn.to_memory_config"(%680) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%680) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %686 = "ttnn.typecast"(%685) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%685) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %687 = "ttnn.repeat"(%686) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%686) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %688 = "ttnn.reshape"(%687) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%687) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %689 = "ttnn.to_memory_config"(%688) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%688) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %690 = "ttnn.multiply"(%682, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%682) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %691 = "ttnn.to_memory_config"(%690) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %692 = "ttnn.multiply"(%684, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%684) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %693 = "ttnn.repeat"(%692) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%692) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %694 = "ttnn.reshape"(%693) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%693) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %695 = "ttnn.permute"(%694) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %696 = "ttnn.to_memory_config"(%695) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%695) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %697 = "ttnn.to_memory_config"(%694) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%694) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %698 = "ttnn.matmul"(%690, %697) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%697) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%690) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %699 = "ttnn.to_memory_config"(%698) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%698) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %700 = "ttnn.add"(%699, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%699) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %701 = "ttnn.to_memory_config"(%700) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%700) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %702 = "ttnn.softmax"(%701) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%701) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %703 = "ttnn.to_memory_config"(%702) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %704 = "ttnn.matmul"(%702, %689) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%702) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %705 = "ttnn.to_memory_config"(%704) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%704) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %706 = "ttnn.typecast"(%705) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%705) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %707 = "ttnn.to_memory_config"(%706) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%706) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %708 = "ttnn.concatenate_heads"(%707) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%707) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %709 = "ttnn.to_memory_config"(%708) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %710 = "ttnn.matmul"(%708, %arg67) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%708) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %711 = "ttnn.add"(%710, %659) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%710) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%659) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %712 = "ttnn.typecast"(%711) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %713 = "ttnn.to_memory_config"(%712) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %714 = "ttnn.pow_scalar"(%712) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %715 = "ttnn.mean"(%714) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%714) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %716 = "ttnn.add"(%715, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%715) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %717 = "ttnn.rsqrt"(%716) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%716) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %718 = "ttnn.to_memory_config"(%717) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %719 = "ttnn.multiply"(%712, %717) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%717) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%712) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %720 = "ttnn.typecast"(%719) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%719) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %721 = "ttnn.to_memory_config"(%720) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%720) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %722 = "ttnn.rms_norm"(%711, %arg72) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %723 = "ttnn.to_memory_config"(%722) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %724 = "ttnn.to_memory_config"(%722) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%722) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %725 = "ttnn.matmul"(%724, %arg68) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %726 = "ttnn.to_memory_config"(%725) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %727 = "ttnn.silu"(%725) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%725) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %728 = "ttnn.to_memory_config"(%727) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %729 = "ttnn.matmul"(%724, %arg69) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%724) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %730 = "ttnn.to_memory_config"(%729) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %731 = "ttnn.multiply"(%727, %729) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%729) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%727) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %732 = "ttnn.to_memory_config"(%731) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %733 = "ttnn.matmul"(%731, %arg70) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%731) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %734 = "ttnn.add"(%733, %711) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%733) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%711) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %735 = "ttnn.to_memory_config"(%734) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%734) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %736 = "ttnn.to_memory_config"(%735) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %737 = "ttnn.typecast"(%736) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%736) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %738 = "ttnn.to_memory_config"(%737) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %739 = "ttnn.pow_scalar"(%737) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %740 = "ttnn.mean"(%739) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%739) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %741 = "ttnn.add"(%740, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%740) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %742 = "ttnn.rsqrt"(%741) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%741) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %743 = "ttnn.to_memory_config"(%742) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %744 = "ttnn.multiply"(%737, %742) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%742) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%737) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %745 = "ttnn.typecast"(%744) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%744) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %746 = "ttnn.to_memory_config"(%745) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%745) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %747 = "ttnn.to_memory_config"(%735) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %748 = "ttnn.rms_norm"(%747, %arg80) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%747) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %749 = "ttnn.to_memory_config"(%748) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %750 = "ttnn.concat"(%arg73, %arg74, %arg75) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %751 = "ttnn.matmul"(%748, %750) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%750) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%748) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %752 = "ttnn.to_memory_config"(%751) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%751) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_21, %key_22, %value_23 = "ttnn.split_query_key_value_and_split_heads"(%752) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%752) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %753 = "ttnn.rotary_embedding"(%query_21, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_21) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %754 = "ttnn.rotary_embedding"(%key_22, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_22) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %755 = "ttnn.reshape"(%754) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%754) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %756 = "ttnn.reshape"(%value_23) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_23) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %757 = "ttnn.to_memory_config"(%753) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%753) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %758 = "ttnn.typecast"(%757) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%757) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %759 = "ttnn.to_memory_config"(%755) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%755) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %760 = "ttnn.typecast"(%759) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%759) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %761 = "ttnn.to_memory_config"(%756) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%756) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %762 = "ttnn.typecast"(%761) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%761) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %763 = "ttnn.repeat"(%762) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%762) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %764 = "ttnn.reshape"(%763) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%763) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %765 = "ttnn.to_memory_config"(%764) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%764) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %766 = "ttnn.multiply"(%758, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%758) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %767 = "ttnn.to_memory_config"(%766) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %768 = "ttnn.multiply"(%760, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%760) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %769 = "ttnn.repeat"(%768) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%768) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %770 = "ttnn.reshape"(%769) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%769) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %771 = "ttnn.permute"(%770) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %772 = "ttnn.to_memory_config"(%771) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%771) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %773 = "ttnn.to_memory_config"(%770) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%770) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %774 = "ttnn.matmul"(%766, %773) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%773) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%766) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %775 = "ttnn.to_memory_config"(%774) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%774) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %776 = "ttnn.add"(%775, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%775) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %777 = "ttnn.to_memory_config"(%776) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%776) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %778 = "ttnn.softmax"(%777) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%777) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %779 = "ttnn.to_memory_config"(%778) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %780 = "ttnn.matmul"(%778, %765) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%778) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %781 = "ttnn.to_memory_config"(%780) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%780) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %782 = "ttnn.typecast"(%781) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%781) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %783 = "ttnn.to_memory_config"(%782) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%782) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %784 = "ttnn.concatenate_heads"(%783) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%783) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %785 = "ttnn.to_memory_config"(%784) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %786 = "ttnn.matmul"(%784, %arg76) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%784) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %787 = "ttnn.add"(%786, %735) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%786) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%735) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %788 = "ttnn.typecast"(%787) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %789 = "ttnn.to_memory_config"(%788) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %790 = "ttnn.pow_scalar"(%788) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %791 = "ttnn.mean"(%790) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%790) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %792 = "ttnn.add"(%791, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%791) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %793 = "ttnn.rsqrt"(%792) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%792) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %794 = "ttnn.to_memory_config"(%793) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %795 = "ttnn.multiply"(%788, %793) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%793) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%788) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %796 = "ttnn.typecast"(%795) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%795) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %797 = "ttnn.to_memory_config"(%796) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%796) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %798 = "ttnn.rms_norm"(%787, %arg81) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %799 = "ttnn.to_memory_config"(%798) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %800 = "ttnn.to_memory_config"(%798) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%798) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %801 = "ttnn.matmul"(%800, %arg77) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %802 = "ttnn.to_memory_config"(%801) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %803 = "ttnn.silu"(%801) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%801) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %804 = "ttnn.to_memory_config"(%803) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %805 = "ttnn.matmul"(%800, %arg78) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%800) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %806 = "ttnn.to_memory_config"(%805) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %807 = "ttnn.multiply"(%803, %805) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%805) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%803) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %808 = "ttnn.to_memory_config"(%807) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %809 = "ttnn.matmul"(%807, %arg79) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%807) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %810 = "ttnn.add"(%809, %787) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%809) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%787) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %811 = "ttnn.to_memory_config"(%810) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%810) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %812 = "ttnn.to_memory_config"(%811) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %813 = "ttnn.typecast"(%812) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%812) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %814 = "ttnn.to_memory_config"(%813) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %815 = "ttnn.pow_scalar"(%813) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %816 = "ttnn.mean"(%815) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%815) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %817 = "ttnn.add"(%816, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%816) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %818 = "ttnn.rsqrt"(%817) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%817) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %819 = "ttnn.to_memory_config"(%818) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %820 = "ttnn.multiply"(%813, %818) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%818) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%813) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %821 = "ttnn.typecast"(%820) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%820) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %822 = "ttnn.to_memory_config"(%821) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%821) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %823 = "ttnn.to_memory_config"(%811) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %824 = "ttnn.rms_norm"(%823, %arg89) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%823) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %825 = "ttnn.to_memory_config"(%824) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %826 = "ttnn.concat"(%arg82, %arg83, %arg84) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %827 = "ttnn.matmul"(%824, %826) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%826) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%824) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %828 = "ttnn.to_memory_config"(%827) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%827) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_24, %key_25, %value_26 = "ttnn.split_query_key_value_and_split_heads"(%828) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%828) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %829 = "ttnn.rotary_embedding"(%query_24, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_24) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %830 = "ttnn.rotary_embedding"(%key_25, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_25) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %831 = "ttnn.reshape"(%830) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%830) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %832 = "ttnn.reshape"(%value_26) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_26) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %833 = "ttnn.to_memory_config"(%829) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%829) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %834 = "ttnn.typecast"(%833) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%833) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %835 = "ttnn.to_memory_config"(%831) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%831) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %836 = "ttnn.typecast"(%835) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%835) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %837 = "ttnn.to_memory_config"(%832) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%832) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %838 = "ttnn.typecast"(%837) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%837) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %839 = "ttnn.repeat"(%838) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%838) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %840 = "ttnn.reshape"(%839) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%839) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %841 = "ttnn.to_memory_config"(%840) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%840) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %842 = "ttnn.multiply"(%834, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%834) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %843 = "ttnn.to_memory_config"(%842) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %844 = "ttnn.multiply"(%836, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%836) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %845 = "ttnn.repeat"(%844) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%844) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %846 = "ttnn.reshape"(%845) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%845) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %847 = "ttnn.permute"(%846) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %848 = "ttnn.to_memory_config"(%847) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%847) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %849 = "ttnn.to_memory_config"(%846) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%846) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %850 = "ttnn.matmul"(%842, %849) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%849) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%842) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %851 = "ttnn.to_memory_config"(%850) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%850) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %852 = "ttnn.add"(%851, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%851) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %853 = "ttnn.to_memory_config"(%852) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%852) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %854 = "ttnn.softmax"(%853) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%853) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %855 = "ttnn.to_memory_config"(%854) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %856 = "ttnn.matmul"(%854, %841) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%854) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %857 = "ttnn.to_memory_config"(%856) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%856) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %858 = "ttnn.typecast"(%857) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%857) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %859 = "ttnn.to_memory_config"(%858) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%858) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %860 = "ttnn.concatenate_heads"(%859) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%859) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %861 = "ttnn.to_memory_config"(%860) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %862 = "ttnn.matmul"(%860, %arg85) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%860) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %863 = "ttnn.add"(%862, %811) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%862) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%811) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %864 = "ttnn.typecast"(%863) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %865 = "ttnn.to_memory_config"(%864) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %866 = "ttnn.pow_scalar"(%864) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %867 = "ttnn.mean"(%866) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%866) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %868 = "ttnn.add"(%867, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%867) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %869 = "ttnn.rsqrt"(%868) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%868) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %870 = "ttnn.to_memory_config"(%869) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %871 = "ttnn.multiply"(%864, %869) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%869) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%864) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %872 = "ttnn.typecast"(%871) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%871) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %873 = "ttnn.to_memory_config"(%872) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%872) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %874 = "ttnn.rms_norm"(%863, %arg90) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %875 = "ttnn.to_memory_config"(%874) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %876 = "ttnn.to_memory_config"(%874) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%874) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %877 = "ttnn.matmul"(%876, %arg86) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %878 = "ttnn.to_memory_config"(%877) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %879 = "ttnn.silu"(%877) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%877) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %880 = "ttnn.to_memory_config"(%879) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %881 = "ttnn.matmul"(%876, %arg87) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%876) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %882 = "ttnn.to_memory_config"(%881) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %883 = "ttnn.multiply"(%879, %881) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%881) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%879) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %884 = "ttnn.to_memory_config"(%883) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %885 = "ttnn.matmul"(%883, %arg88) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%883) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %886 = "ttnn.add"(%885, %863) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%885) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%863) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %887 = "ttnn.to_memory_config"(%886) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%886) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %888 = "ttnn.to_memory_config"(%887) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %889 = "ttnn.typecast"(%888) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%888) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %890 = "ttnn.to_memory_config"(%889) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %891 = "ttnn.pow_scalar"(%889) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %892 = "ttnn.mean"(%891) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%891) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %893 = "ttnn.add"(%892, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%892) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %894 = "ttnn.rsqrt"(%893) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%893) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %895 = "ttnn.to_memory_config"(%894) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %896 = "ttnn.multiply"(%889, %894) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%894) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%889) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %897 = "ttnn.typecast"(%896) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%896) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %898 = "ttnn.to_memory_config"(%897) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%897) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %899 = "ttnn.to_memory_config"(%887) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %900 = "ttnn.rms_norm"(%899, %arg98) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%899) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %901 = "ttnn.to_memory_config"(%900) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %902 = "ttnn.concat"(%arg91, %arg92, %arg93) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %903 = "ttnn.matmul"(%900, %902) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%902) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%900) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %904 = "ttnn.to_memory_config"(%903) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%903) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_27, %key_28, %value_29 = "ttnn.split_query_key_value_and_split_heads"(%904) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%904) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %905 = "ttnn.rotary_embedding"(%query_27, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_27) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %906 = "ttnn.rotary_embedding"(%key_28, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_28) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %907 = "ttnn.reshape"(%906) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%906) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %908 = "ttnn.reshape"(%value_29) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_29) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %909 = "ttnn.to_memory_config"(%905) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%905) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %910 = "ttnn.typecast"(%909) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%909) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %911 = "ttnn.to_memory_config"(%907) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%907) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %912 = "ttnn.typecast"(%911) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%911) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %913 = "ttnn.to_memory_config"(%908) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%908) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %914 = "ttnn.typecast"(%913) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%913) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %915 = "ttnn.repeat"(%914) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%914) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %916 = "ttnn.reshape"(%915) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%915) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %917 = "ttnn.to_memory_config"(%916) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%916) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %918 = "ttnn.multiply"(%910, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%910) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %919 = "ttnn.to_memory_config"(%918) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %920 = "ttnn.multiply"(%912, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%912) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %921 = "ttnn.repeat"(%920) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%920) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %922 = "ttnn.reshape"(%921) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%921) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %923 = "ttnn.permute"(%922) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %924 = "ttnn.to_memory_config"(%923) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%923) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %925 = "ttnn.to_memory_config"(%922) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%922) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %926 = "ttnn.matmul"(%918, %925) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%925) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%918) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %927 = "ttnn.to_memory_config"(%926) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%926) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %928 = "ttnn.add"(%927, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%927) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %929 = "ttnn.to_memory_config"(%928) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%928) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %930 = "ttnn.softmax"(%929) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%929) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %931 = "ttnn.to_memory_config"(%930) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %932 = "ttnn.matmul"(%930, %917) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%930) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %933 = "ttnn.to_memory_config"(%932) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%932) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %934 = "ttnn.typecast"(%933) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%933) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %935 = "ttnn.to_memory_config"(%934) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%934) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %936 = "ttnn.concatenate_heads"(%935) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%935) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %937 = "ttnn.to_memory_config"(%936) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %938 = "ttnn.matmul"(%936, %arg94) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%936) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %939 = "ttnn.add"(%938, %887) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%938) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%887) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %940 = "ttnn.typecast"(%939) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %941 = "ttnn.to_memory_config"(%940) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %942 = "ttnn.pow_scalar"(%940) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %943 = "ttnn.mean"(%942) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%942) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %944 = "ttnn.add"(%943, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%943) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %945 = "ttnn.rsqrt"(%944) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%944) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %946 = "ttnn.to_memory_config"(%945) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %947 = "ttnn.multiply"(%940, %945) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%945) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%940) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %948 = "ttnn.typecast"(%947) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%947) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %949 = "ttnn.to_memory_config"(%948) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%948) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %950 = "ttnn.rms_norm"(%939, %arg99) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %951 = "ttnn.to_memory_config"(%950) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %952 = "ttnn.to_memory_config"(%950) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%950) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %953 = "ttnn.matmul"(%952, %arg95) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %954 = "ttnn.to_memory_config"(%953) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %955 = "ttnn.silu"(%953) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%953) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %956 = "ttnn.to_memory_config"(%955) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %957 = "ttnn.matmul"(%952, %arg96) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%952) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %958 = "ttnn.to_memory_config"(%957) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %959 = "ttnn.multiply"(%955, %957) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%957) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%955) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %960 = "ttnn.to_memory_config"(%959) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %961 = "ttnn.matmul"(%959, %arg97) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%959) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %962 = "ttnn.add"(%961, %939) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%961) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%939) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %963 = "ttnn.to_memory_config"(%962) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%962) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %964 = "ttnn.to_memory_config"(%963) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %965 = "ttnn.typecast"(%964) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%964) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %966 = "ttnn.to_memory_config"(%965) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %967 = "ttnn.pow_scalar"(%965) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %968 = "ttnn.mean"(%967) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%967) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %969 = "ttnn.add"(%968, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%968) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %970 = "ttnn.rsqrt"(%969) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%969) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %971 = "ttnn.to_memory_config"(%970) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %972 = "ttnn.multiply"(%965, %970) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%970) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%965) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %973 = "ttnn.typecast"(%972) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%972) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %974 = "ttnn.to_memory_config"(%973) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%973) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %975 = "ttnn.to_memory_config"(%963) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %976 = "ttnn.rms_norm"(%975, %arg107) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%975) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %977 = "ttnn.to_memory_config"(%976) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %978 = "ttnn.concat"(%arg100, %arg101, %arg102) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %979 = "ttnn.matmul"(%976, %978) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%978) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%976) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %980 = "ttnn.to_memory_config"(%979) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%979) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_30, %key_31, %value_32 = "ttnn.split_query_key_value_and_split_heads"(%980) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%980) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %981 = "ttnn.rotary_embedding"(%query_30, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_30) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %982 = "ttnn.rotary_embedding"(%key_31, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_31) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %983 = "ttnn.reshape"(%982) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%982) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %984 = "ttnn.reshape"(%value_32) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_32) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %985 = "ttnn.to_memory_config"(%981) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%981) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %986 = "ttnn.typecast"(%985) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%985) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %987 = "ttnn.to_memory_config"(%983) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%983) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %988 = "ttnn.typecast"(%987) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%987) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %989 = "ttnn.to_memory_config"(%984) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%984) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %990 = "ttnn.typecast"(%989) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%989) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %991 = "ttnn.repeat"(%990) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%990) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %992 = "ttnn.reshape"(%991) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%991) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %993 = "ttnn.to_memory_config"(%992) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%992) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %994 = "ttnn.multiply"(%986, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%986) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %995 = "ttnn.to_memory_config"(%994) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %996 = "ttnn.multiply"(%988, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%988) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %997 = "ttnn.repeat"(%996) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%996) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %998 = "ttnn.reshape"(%997) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%997) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %999 = "ttnn.permute"(%998) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %1000 = "ttnn.to_memory_config"(%999) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%999) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %1001 = "ttnn.to_memory_config"(%998) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%998) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1002 = "ttnn.matmul"(%994, %1001) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1001) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%994) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1003 = "ttnn.to_memory_config"(%1002) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1002) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1004 = "ttnn.add"(%1003, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1003) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1005 = "ttnn.to_memory_config"(%1004) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1004) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1006 = "ttnn.softmax"(%1005) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1005) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1007 = "ttnn.to_memory_config"(%1006) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %1008 = "ttnn.matmul"(%1006, %993) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1006) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1009 = "ttnn.to_memory_config"(%1008) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1008) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1010 = "ttnn.typecast"(%1009) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1009) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1011 = "ttnn.to_memory_config"(%1010) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%1010) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1012 = "ttnn.concatenate_heads"(%1011) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%1011) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1013 = "ttnn.to_memory_config"(%1012) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1014 = "ttnn.matmul"(%1012, %arg103) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1012) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %1015 = "ttnn.add"(%1014, %963) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1014) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%963) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1016 = "ttnn.typecast"(%1015) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1017 = "ttnn.to_memory_config"(%1016) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1018 = "ttnn.pow_scalar"(%1016) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1019 = "ttnn.mean"(%1018) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1018) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1020 = "ttnn.add"(%1019, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1019) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1021 = "ttnn.rsqrt"(%1020) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1020) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1022 = "ttnn.to_memory_config"(%1021) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1023 = "ttnn.multiply"(%1016, %1021) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1021) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%1016) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1024 = "ttnn.typecast"(%1023) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1023) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1025 = "ttnn.to_memory_config"(%1024) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1024) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1026 = "ttnn.rms_norm"(%1015, %arg108) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %1027 = "ttnn.to_memory_config"(%1026) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1028 = "ttnn.to_memory_config"(%1026) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1026) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1029 = "ttnn.matmul"(%1028, %arg104) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %1030 = "ttnn.to_memory_config"(%1029) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1031 = "ttnn.silu"(%1029) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1029) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1032 = "ttnn.to_memory_config"(%1031) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1033 = "ttnn.matmul"(%1028, %arg105) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1028) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1034 = "ttnn.to_memory_config"(%1033) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1035 = "ttnn.multiply"(%1031, %1033) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1033) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%1031) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1036 = "ttnn.to_memory_config"(%1035) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1037 = "ttnn.matmul"(%1035, %arg106) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1035) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1038 = "ttnn.add"(%1037, %1015) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1037) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%1015) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1039 = "ttnn.to_memory_config"(%1038) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1038) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1040 = "ttnn.to_memory_config"(%1039) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %1041 = "ttnn.typecast"(%1040) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1040) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1042 = "ttnn.to_memory_config"(%1041) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1043 = "ttnn.pow_scalar"(%1041) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %1044 = "ttnn.mean"(%1043) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1043) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1045 = "ttnn.add"(%1044, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1044) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1046 = "ttnn.rsqrt"(%1045) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1045) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1047 = "ttnn.to_memory_config"(%1046) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1048 = "ttnn.multiply"(%1041, %1046) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1046) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%1041) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1049 = "ttnn.typecast"(%1048) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1048) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1050 = "ttnn.to_memory_config"(%1049) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1049) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1051 = "ttnn.to_memory_config"(%1039) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %1052 = "ttnn.rms_norm"(%1051, %arg116) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1051) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1053 = "ttnn.to_memory_config"(%1052) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1054 = "ttnn.concat"(%arg109, %arg110, %arg111) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %1055 = "ttnn.matmul"(%1052, %1054) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%1054) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%1052) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1056 = "ttnn.to_memory_config"(%1055) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%1055) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_33, %key_34, %value_35 = "ttnn.split_query_key_value_and_split_heads"(%1056) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%1056) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %1057 = "ttnn.rotary_embedding"(%query_33, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_33) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1058 = "ttnn.rotary_embedding"(%key_34, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_34) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1059 = "ttnn.reshape"(%1058) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%1058) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1060 = "ttnn.reshape"(%value_35) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_35) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1061 = "ttnn.to_memory_config"(%1057) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1057) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1062 = "ttnn.typecast"(%1061) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1061) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1063 = "ttnn.to_memory_config"(%1059) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%1059) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %1064 = "ttnn.typecast"(%1063) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1063) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %1065 = "ttnn.to_memory_config"(%1060) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%1060) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %1066 = "ttnn.typecast"(%1065) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1065) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %1067 = "ttnn.repeat"(%1066) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%1066) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1068 = "ttnn.reshape"(%1067) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1067) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %1069 = "ttnn.to_memory_config"(%1068) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1068) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1070 = "ttnn.multiply"(%1062, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1062) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1071 = "ttnn.to_memory_config"(%1070) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %1072 = "ttnn.multiply"(%1064, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1064) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1073 = "ttnn.repeat"(%1072) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%1072) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1074 = "ttnn.reshape"(%1073) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1073) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %1075 = "ttnn.permute"(%1074) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %1076 = "ttnn.to_memory_config"(%1075) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%1075) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %1077 = "ttnn.to_memory_config"(%1074) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1074) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1078 = "ttnn.matmul"(%1070, %1077) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1077) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%1070) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1079 = "ttnn.to_memory_config"(%1078) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1078) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1080 = "ttnn.add"(%1079, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1079) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1081 = "ttnn.to_memory_config"(%1080) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1080) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1082 = "ttnn.softmax"(%1081) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1081) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1083 = "ttnn.to_memory_config"(%1082) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %1084 = "ttnn.matmul"(%1082, %1069) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1082) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1085 = "ttnn.to_memory_config"(%1084) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1084) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1086 = "ttnn.typecast"(%1085) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1085) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1087 = "ttnn.to_memory_config"(%1086) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%1086) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1088 = "ttnn.concatenate_heads"(%1087) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%1087) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1089 = "ttnn.to_memory_config"(%1088) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1090 = "ttnn.matmul"(%1088, %arg112) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1088) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %1091 = "ttnn.add"(%1090, %1039) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1090) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%1039) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1092 = "ttnn.typecast"(%1091) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1093 = "ttnn.to_memory_config"(%1092) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1094 = "ttnn.pow_scalar"(%1092) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1095 = "ttnn.mean"(%1094) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1094) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1096 = "ttnn.add"(%1095, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1095) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1097 = "ttnn.rsqrt"(%1096) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1096) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1098 = "ttnn.to_memory_config"(%1097) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1099 = "ttnn.multiply"(%1092, %1097) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1097) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%1092) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1100 = "ttnn.typecast"(%1099) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1099) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1101 = "ttnn.to_memory_config"(%1100) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1100) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1102 = "ttnn.rms_norm"(%1091, %arg117) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %1103 = "ttnn.to_memory_config"(%1102) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1104 = "ttnn.to_memory_config"(%1102) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1102) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1105 = "ttnn.matmul"(%1104, %arg113) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %1106 = "ttnn.to_memory_config"(%1105) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1107 = "ttnn.silu"(%1105) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1105) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1108 = "ttnn.to_memory_config"(%1107) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1109 = "ttnn.matmul"(%1104, %arg114) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1104) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1110 = "ttnn.to_memory_config"(%1109) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1111 = "ttnn.multiply"(%1107, %1109) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1109) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%1107) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1112 = "ttnn.to_memory_config"(%1111) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1113 = "ttnn.matmul"(%1111, %arg115) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1111) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1114 = "ttnn.add"(%1113, %1091) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1113) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%1091) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1115 = "ttnn.to_memory_config"(%1114) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1114) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1116 = "ttnn.to_memory_config"(%1115) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %1117 = "ttnn.typecast"(%1116) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1116) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1118 = "ttnn.to_memory_config"(%1117) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1119 = "ttnn.pow_scalar"(%1117) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %1120 = "ttnn.mean"(%1119) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1119) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1121 = "ttnn.add"(%1120, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1120) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1122 = "ttnn.rsqrt"(%1121) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1121) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1123 = "ttnn.to_memory_config"(%1122) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1124 = "ttnn.multiply"(%1117, %1122) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1122) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%1117) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1125 = "ttnn.typecast"(%1124) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1124) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1126 = "ttnn.to_memory_config"(%1125) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1125) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1127 = "ttnn.to_memory_config"(%1115) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %1128 = "ttnn.rms_norm"(%1127, %arg125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1127) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1129 = "ttnn.to_memory_config"(%1128) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1130 = "ttnn.concat"(%arg118, %arg119, %arg120) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %1131 = "ttnn.matmul"(%1128, %1130) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%1130) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%1128) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1132 = "ttnn.to_memory_config"(%1131) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%1131) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_36, %key_37, %value_38 = "ttnn.split_query_key_value_and_split_heads"(%1132) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%1132) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %1133 = "ttnn.rotary_embedding"(%query_36, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_36) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1134 = "ttnn.rotary_embedding"(%key_37, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_37) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1135 = "ttnn.reshape"(%1134) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%1134) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1136 = "ttnn.reshape"(%value_38) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_38) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1137 = "ttnn.to_memory_config"(%1133) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1133) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1138 = "ttnn.typecast"(%1137) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1137) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1139 = "ttnn.to_memory_config"(%1135) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%1135) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %1140 = "ttnn.typecast"(%1139) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1139) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %1141 = "ttnn.to_memory_config"(%1136) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%1136) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %1142 = "ttnn.typecast"(%1141) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1141) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %1143 = "ttnn.repeat"(%1142) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%1142) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1144 = "ttnn.reshape"(%1143) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1143) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %1145 = "ttnn.to_memory_config"(%1144) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1144) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1146 = "ttnn.multiply"(%1138, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1138) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1147 = "ttnn.to_memory_config"(%1146) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %1148 = "ttnn.multiply"(%1140, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1140) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1149 = "ttnn.repeat"(%1148) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%1148) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1150 = "ttnn.reshape"(%1149) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1149) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %1151 = "ttnn.permute"(%1150) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %1152 = "ttnn.to_memory_config"(%1151) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%1151) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %1153 = "ttnn.to_memory_config"(%1150) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1150) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1154 = "ttnn.matmul"(%1146, %1153) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1153) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%1146) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1155 = "ttnn.to_memory_config"(%1154) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1154) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1156 = "ttnn.add"(%1155, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1155) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1157 = "ttnn.to_memory_config"(%1156) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1156) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1158 = "ttnn.softmax"(%1157) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1157) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1159 = "ttnn.to_memory_config"(%1158) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %1160 = "ttnn.matmul"(%1158, %1145) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1158) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1161 = "ttnn.to_memory_config"(%1160) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1160) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1162 = "ttnn.typecast"(%1161) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1161) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1163 = "ttnn.to_memory_config"(%1162) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%1162) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1164 = "ttnn.concatenate_heads"(%1163) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%1163) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1165 = "ttnn.to_memory_config"(%1164) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1166 = "ttnn.matmul"(%1164, %arg121) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1164) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %1167 = "ttnn.add"(%1166, %1115) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1166) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%1115) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1168 = "ttnn.typecast"(%1167) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1169 = "ttnn.to_memory_config"(%1168) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1170 = "ttnn.pow_scalar"(%1168) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1171 = "ttnn.mean"(%1170) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1170) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1172 = "ttnn.add"(%1171, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1171) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1173 = "ttnn.rsqrt"(%1172) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1172) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1174 = "ttnn.to_memory_config"(%1173) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1175 = "ttnn.multiply"(%1168, %1173) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1173) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%1168) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1176 = "ttnn.typecast"(%1175) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1175) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1177 = "ttnn.to_memory_config"(%1176) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1176) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1178 = "ttnn.rms_norm"(%1167, %arg126) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %1179 = "ttnn.to_memory_config"(%1178) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1180 = "ttnn.to_memory_config"(%1178) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1178) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1181 = "ttnn.matmul"(%1180, %arg122) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %1182 = "ttnn.to_memory_config"(%1181) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1183 = "ttnn.silu"(%1181) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1181) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1184 = "ttnn.to_memory_config"(%1183) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1185 = "ttnn.matmul"(%1180, %arg123) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1180) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1186 = "ttnn.to_memory_config"(%1185) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1187 = "ttnn.multiply"(%1183, %1185) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1185) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%1183) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1188 = "ttnn.to_memory_config"(%1187) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1189 = "ttnn.matmul"(%1187, %arg124) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1187) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1190 = "ttnn.add"(%1189, %1167) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1189) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%1167) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1191 = "ttnn.to_memory_config"(%1190) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1190) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1192 = "ttnn.to_memory_config"(%1191) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %1193 = "ttnn.typecast"(%1192) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1192) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1194 = "ttnn.to_memory_config"(%1193) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1195 = "ttnn.pow_scalar"(%1193) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %1196 = "ttnn.mean"(%1195) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1195) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1197 = "ttnn.add"(%1196, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1196) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1198 = "ttnn.rsqrt"(%1197) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1197) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1199 = "ttnn.to_memory_config"(%1198) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1200 = "ttnn.multiply"(%1193, %1198) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1198) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%1193) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1201 = "ttnn.typecast"(%1200) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1200) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1202 = "ttnn.to_memory_config"(%1201) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1201) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1203 = "ttnn.to_memory_config"(%1191) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %1204 = "ttnn.rms_norm"(%1203, %arg134) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1203) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1205 = "ttnn.to_memory_config"(%1204) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1206 = "ttnn.concat"(%arg127, %arg128, %arg129) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %1207 = "ttnn.matmul"(%1204, %1206) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%1206) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%1204) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1208 = "ttnn.to_memory_config"(%1207) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%1207) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_39, %key_40, %value_41 = "ttnn.split_query_key_value_and_split_heads"(%1208) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%1208) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %1209 = "ttnn.rotary_embedding"(%query_39, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_39) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1210 = "ttnn.rotary_embedding"(%key_40, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_40) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1211 = "ttnn.reshape"(%1210) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%1210) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1212 = "ttnn.reshape"(%value_41) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_41) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1213 = "ttnn.to_memory_config"(%1209) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1209) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1214 = "ttnn.typecast"(%1213) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1213) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1215 = "ttnn.to_memory_config"(%1211) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%1211) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %1216 = "ttnn.typecast"(%1215) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1215) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %1217 = "ttnn.to_memory_config"(%1212) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%1212) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %1218 = "ttnn.typecast"(%1217) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1217) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %1219 = "ttnn.repeat"(%1218) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%1218) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1220 = "ttnn.reshape"(%1219) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1219) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %1221 = "ttnn.to_memory_config"(%1220) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1220) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1222 = "ttnn.multiply"(%1214, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1214) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1223 = "ttnn.to_memory_config"(%1222) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %1224 = "ttnn.multiply"(%1216, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1216) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1225 = "ttnn.repeat"(%1224) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%1224) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1226 = "ttnn.reshape"(%1225) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1225) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %1227 = "ttnn.permute"(%1226) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %1228 = "ttnn.to_memory_config"(%1227) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%1227) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %1229 = "ttnn.to_memory_config"(%1226) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1226) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1230 = "ttnn.matmul"(%1222, %1229) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1229) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%1222) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1231 = "ttnn.to_memory_config"(%1230) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1230) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1232 = "ttnn.add"(%1231, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1231) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1233 = "ttnn.to_memory_config"(%1232) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1232) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1234 = "ttnn.softmax"(%1233) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1233) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1235 = "ttnn.to_memory_config"(%1234) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %1236 = "ttnn.matmul"(%1234, %1221) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1234) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1237 = "ttnn.to_memory_config"(%1236) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1236) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1238 = "ttnn.typecast"(%1237) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1237) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1239 = "ttnn.to_memory_config"(%1238) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%1238) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1240 = "ttnn.concatenate_heads"(%1239) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%1239) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1241 = "ttnn.to_memory_config"(%1240) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1242 = "ttnn.matmul"(%1240, %arg130) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1240) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %1243 = "ttnn.add"(%1242, %1191) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1242) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%1191) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1244 = "ttnn.typecast"(%1243) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1245 = "ttnn.to_memory_config"(%1244) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1246 = "ttnn.pow_scalar"(%1244) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1247 = "ttnn.mean"(%1246) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1246) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1248 = "ttnn.add"(%1247, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1247) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1249 = "ttnn.rsqrt"(%1248) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1248) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1250 = "ttnn.to_memory_config"(%1249) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1251 = "ttnn.multiply"(%1244, %1249) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1249) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%1244) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1252 = "ttnn.typecast"(%1251) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1251) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1253 = "ttnn.to_memory_config"(%1252) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1252) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1254 = "ttnn.rms_norm"(%1243, %arg135) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %1255 = "ttnn.to_memory_config"(%1254) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1256 = "ttnn.to_memory_config"(%1254) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1254) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1257 = "ttnn.matmul"(%1256, %arg131) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %1258 = "ttnn.to_memory_config"(%1257) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1259 = "ttnn.silu"(%1257) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1257) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1260 = "ttnn.to_memory_config"(%1259) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1261 = "ttnn.matmul"(%1256, %arg132) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1256) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1262 = "ttnn.to_memory_config"(%1261) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1263 = "ttnn.multiply"(%1259, %1261) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1261) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%1259) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1264 = "ttnn.to_memory_config"(%1263) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1265 = "ttnn.matmul"(%1263, %arg133) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1263) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1266 = "ttnn.add"(%1265, %1243) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1265) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%1243) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1267 = "ttnn.to_memory_config"(%1266) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1266) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1268 = "ttnn.to_memory_config"(%1267) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %1269 = "ttnn.typecast"(%1268) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1268) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1270 = "ttnn.to_memory_config"(%1269) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1271 = "ttnn.pow_scalar"(%1269) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %1272 = "ttnn.mean"(%1271) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1271) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1273 = "ttnn.add"(%1272, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1272) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1274 = "ttnn.rsqrt"(%1273) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1273) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1275 = "ttnn.to_memory_config"(%1274) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1276 = "ttnn.multiply"(%1269, %1274) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1274) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%1269) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1277 = "ttnn.typecast"(%1276) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1276) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1278 = "ttnn.to_memory_config"(%1277) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1277) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1279 = "ttnn.to_memory_config"(%1267) : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        %1280 = "ttnn.rms_norm"(%1279, %arg143) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1279) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1281 = "ttnn.to_memory_config"(%1280) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1282 = "ttnn.concat"(%arg136, %arg137, %arg138) <{dim = 0 : si32}> : (tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>) -> tensor<3072x2048xbf16, #ttnn_layout68>
        %1283 = "ttnn.matmul"(%1280, %1282) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 12, per_core_m = 2, per_core_n = 12, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<3072x2048xbf16, #ttnn_layout68>) -> tensor<1x512x3072xbf16, #ttnn_layout69>
        "ttnn.deallocate"(%1282) <{force = false}> : (tensor<3072x2048xbf16, #ttnn_layout68>) -> ()
        "ttnn.deallocate"(%1280) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1284 = "ttnn.to_memory_config"(%1283) : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> tensor<1x512x3072xbf16, #ttnn_layout70>
        "ttnn.deallocate"(%1283) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout69>) -> ()
        %query_42, %key_43, %value_44 = "ttnn.split_query_key_value_and_split_heads"(%1284) <{num_heads = 32 : ui32, num_kv_heads = 8 : ui32, transpose_key = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x8x512x64xbf16, #ttnn_layout72>)
        "ttnn.deallocate"(%1284) <{force = false}> : (tensor<1x512x3072xbf16, #ttnn_layout70>) -> ()
        %1285 = "ttnn.rotary_embedding"(%query_42, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%query_42) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1286 = "ttnn.rotary_embedding"(%key_43, %121, %125) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>) -> tensor<1x8x512x64xbf16, #ttnn_layout72>
        "ttnn.deallocate"(%key_43) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        "ttnn.deallocate"(%125) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout16>) -> ()
        "ttnn.deallocate"(%121) <{force = false}> : (tensor<1x1x512x64xbf16, #ttnn_layout16>) -> ()
        %1287 = "ttnn.reshape"(%1286) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%1286) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1288 = "ttnn.reshape"(%value_44) <{shape = [1 : i32, 8 : i32, 1 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout74>
        "ttnn.deallocate"(%value_44) <{force = false}> : (tensor<1x8x512x64xbf16, #ttnn_layout72>) -> ()
        %1289 = "ttnn.to_memory_config"(%1285) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1285) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1290 = "ttnn.typecast"(%1289) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1289) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1291 = "ttnn.to_memory_config"(%1287) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%1287) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %1292 = "ttnn.typecast"(%1291) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1291) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %1293 = "ttnn.to_memory_config"(%1288) : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> tensor<1x8x1x512x64xbf16, #ttnn_layout77>
        "ttnn.deallocate"(%1288) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout74>) -> ()
        %1294 = "ttnn.typecast"(%1293) : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1293) <{force = false}> : (tensor<1x8x1x512x64xbf16, #ttnn_layout77>) -> ()
        %1295 = "ttnn.repeat"(%1294) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%1294) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1296 = "ttnn.reshape"(%1295) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1295) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %1297 = "ttnn.to_memory_config"(%1296) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1296) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1298 = "ttnn.multiply"(%1290, %149) : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x1x1x1xf32, #ttnn_layout23>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1290) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        "ttnn.deallocate"(%149) <{force = false}> : (tensor<1x1x1x1xf32, #ttnn_layout23>) -> ()
        %1299 = "ttnn.to_memory_config"(%1298) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        %1300 = "ttnn.multiply"(%1292, %154) : (tensor<1x8x1x512x64xf32, #ttnn_layout78>, tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> tensor<1x8x1x512x64xf32, #ttnn_layout78>
        "ttnn.deallocate"(%1292) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        "ttnn.deallocate"(%154) <{force = false}> : (tensor<1x1x1x1x1xf32, #ttnn_layout21>) -> ()
        %1301 = "ttnn.repeat"(%1300) <{repeat_dims = #ttnn.shape<1x1x4x1x1>}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> tensor<1x8x4x512x64xf32, #ttnn_layout79>
        "ttnn.deallocate"(%1300) <{force = false}> : (tensor<1x8x1x512x64xf32, #ttnn_layout78>) -> ()
        %1302 = "ttnn.reshape"(%1301) <{shape = [1 : i32, 32 : i32, 512 : i32, 64 : i32]}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1301) <{force = false}> : (tensor<1x8x4x512x64xf32, #ttnn_layout79>) -> ()
        %1303 = "ttnn.permute"(%1302) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x64x512xf32, #ttnn_layout81>
        %1304 = "ttnn.to_memory_config"(%1303) : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> tensor<1x32x64x512xf32, #ttnn_layout18>
        "ttnn.deallocate"(%1303) <{force = false}> : (tensor<1x32x64x512xf32, #ttnn_layout81>) -> ()
        %1305 = "ttnn.to_memory_config"(%1302) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1302) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1306 = "ttnn.matmul"(%1298, %1305) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x32x512x64xf32, #ttnn_layout76>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1305) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        "ttnn.deallocate"(%1298) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1307 = "ttnn.to_memory_config"(%1306) : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1306) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1308 = "ttnn.add"(%1307, %162) : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x1x512x512xf32, #ttnn_layout83>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1307) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        "ttnn.deallocate"(%162) <{force = false}> : (tensor<1x1x512x512xf32, #ttnn_layout83>) -> ()
        %1309 = "ttnn.to_memory_config"(%1308) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        "ttnn.deallocate"(%1308) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1310 = "ttnn.softmax"(%1309) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> tensor<1x32x512x512xf32, #ttnn_layout84>
        "ttnn.deallocate"(%1309) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout19>) -> ()
        %1311 = "ttnn.to_memory_config"(%1310) : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> tensor<1x32x512x512xf32, #ttnn_layout19>
        %1312 = "ttnn.matmul"(%1310, %1297) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>, tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout17>
        "ttnn.deallocate"(%1310) <{force = false}> : (tensor<1x32x512x512xf32, #ttnn_layout84>) -> ()
        %1313 = "ttnn.to_memory_config"(%1312) : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> tensor<1x32x512x64xf32, #ttnn_layout76>
        "ttnn.deallocate"(%1312) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout17>) -> ()
        %1314 = "ttnn.typecast"(%1313) : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> tensor<1x32x512x64xbf16, #ttnn_layout75>
        "ttnn.deallocate"(%1313) <{force = false}> : (tensor<1x32x512x64xf32, #ttnn_layout76>) -> ()
        %1315 = "ttnn.to_memory_config"(%1314) : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> tensor<1x32x512x64xbf16, #ttnn_layout71>
        "ttnn.deallocate"(%1314) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout75>) -> ()
        %1316 = "ttnn.concatenate_heads"(%1315) : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%1315) <{force = false}> : (tensor<1x32x512x64xbf16, #ttnn_layout71>) -> ()
        %1317 = "ttnn.to_memory_config"(%1316) : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1318 = "ttnn.matmul"(%1316, %arg139) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_1d_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 8, out_subblock_h = 1, out_subblock_w = 1, out_block_h = 16, out_block_w = 1, per_core_m = 16, per_core_n = 1, fuse_batch = true, mcast_in0 = true, gather_in0 = false, hop_cores = #ttnn.core_range_set<>, num_global_cb_receivers = 0, untilize_out = false>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout28>, tensor<2048x2048xbf16, #ttnn_layout1>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1316) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %1319 = "ttnn.add"(%1318, %1267) : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<1x512x2048xbf16, #ttnn_layout15>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1318) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        "ttnn.deallocate"(%1267) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout15>) -> ()
        %1320 = "ttnn.typecast"(%1319) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1321 = "ttnn.to_memory_config"(%1320) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1322 = "ttnn.pow_scalar"(%1320) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        %1323 = "ttnn.mean"(%1322) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1322) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1324 = "ttnn.add"(%1323, %105) : (tensor<1x512x1xf32, #ttnn_layout64>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1323) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1325 = "ttnn.rsqrt"(%1324) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1324) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        %1326 = "ttnn.to_memory_config"(%1325) : (tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1327 = "ttnn.multiply"(%1320, %1325) : (tensor<1x512x2048xf32, #ttnn_layout64>, tensor<1x512x1xf32, #ttnn_layout64>) -> tensor<1x512x2048xf32, #ttnn_layout64>
        "ttnn.deallocate"(%1325) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout64>) -> ()
        "ttnn.deallocate"(%1320) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1328 = "ttnn.typecast"(%1327) : (tensor<1x512x2048xf32, #ttnn_layout64>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        "ttnn.deallocate"(%1327) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout64>) -> ()
        %1329 = "ttnn.to_memory_config"(%1328) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1328) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1330 = "ttnn.rms_norm"(%1319, %arg144) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout63>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout63>
        %1331 = "ttnn.to_memory_config"(%1330) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1332 = "ttnn.to_memory_config"(%1330) : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1330) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1333 = "ttnn.matmul"(%1332, %arg140) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        %1334 = "ttnn.to_memory_config"(%1333) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1335 = "ttnn.silu"(%1333) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1333) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1336 = "ttnn.to_memory_config"(%1335) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1337 = "ttnn.matmul"(%1332, %arg141) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 32, per_core_m = 2, per_core_n = 32, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<8192x2048xbf16, #ttnn_layout3>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1332) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1338 = "ttnn.to_memory_config"(%1337) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1339 = "ttnn.multiply"(%1335, %1337) : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout85>
        "ttnn.deallocate"(%1337) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        "ttnn.deallocate"(%1335) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1340 = "ttnn.to_memory_config"(%1339) : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> tensor<1x512x8192xbf16, #ttnn_layout20>
        %1341 = "ttnn.matmul"(%1339, %arg142) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, in0_block_w = 2, out_subblock_h = 1, out_subblock_w = 4, out_block_h = 2, out_block_w = 8, per_core_m = 2, per_core_n = 8, transpose_mcast = false, fuse_batch = true>, transpose_a = false, transpose_b = true}> : (tensor<1x512x8192xbf16, #ttnn_layout85>, tensor<2048x8192xbf16, #ttnn_layout4>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1339) <{force = false}> : (tensor<1x512x8192xbf16, #ttnn_layout85>) -> ()
        %1342 = "ttnn.add"(%1341, %1319) : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<1x512x2048xbf16, #ttnn_layout63>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1341) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        "ttnn.deallocate"(%1319) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout63>) -> ()
        %1343 = "ttnn.typecast"(%1342) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %1344 = "ttnn.to_memory_config"(%1343) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout14>
        %1345 = "ttnn.pow_scalar"(%1343) <{rhs = 2.000000e+00 : f32}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        %1346 = "ttnn.mean"(%1345) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1345) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1347 = "ttnn.add"(%1346, %105) : (tensor<1x512x1xf32, #ttnn_layout66>, tensor<1x1x1xf32, #ttnn_layout10>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1346) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%105) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout10>) -> ()
        %1348 = "ttnn.rsqrt"(%1347) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout66>
        "ttnn.deallocate"(%1347) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        %1349 = "ttnn.to_memory_config"(%1348) : (tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x1xf32, #ttnn_layout13>
        %1350 = "ttnn.multiply"(%1343, %1348) : (tensor<1x512x2048xf32, #ttnn_layout65>, tensor<1x512x1xf32, #ttnn_layout66>) -> tensor<1x512x2048xf32, #ttnn_layout65>
        "ttnn.deallocate"(%1348) <{force = false}> : (tensor<1x512x1xf32, #ttnn_layout66>) -> ()
        "ttnn.deallocate"(%1343) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1351 = "ttnn.typecast"(%1350) : (tensor<1x512x2048xf32, #ttnn_layout65>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1350) <{force = false}> : (tensor<1x512x2048xf32, #ttnn_layout65>) -> ()
        %1352 = "ttnn.to_memory_config"(%1351) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        "ttnn.deallocate"(%1351) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1353 = "ttnn.rms_norm"(%1342, %arg145) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, math_approx_mode = false, fp32_dest_acc_en = true, packer_l1_acc = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x512x2048xbf16, #ttnn_layout67>, tensor<2048xbf16, #ttnn_layout5>) -> tensor<1x512x2048xbf16, #ttnn_layout67>
        "ttnn.deallocate"(%1342) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1354 = "ttnn.to_memory_config"(%1353) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout15>
        %1355 = "ttnn.to_memory_config"(%1353) : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> tensor<1x512x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%1353) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout67>) -> ()
        %1356 = "ttnn.slice_static"(%1355) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 511 : i32, 2048 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> tensor<1x511x2048xbf16, #ttnn_layout28>
        "ttnn.deallocate"(%1355) <{force = false}> : (tensor<1x512x2048xbf16, #ttnn_layout28>) -> ()
        %1357 = "ttnn.matmul"(%1356, %arg0) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = true}> : (tensor<1x511x2048xbf16, #ttnn_layout28>, tensor<128256x2048xbf16, #ttnn_layout>) -> tensor<1x511x128256xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%1356) <{force = false}> : (tensor<1x511x2048xbf16, #ttnn_layout28>) -> ()
        %1358 = "ttnn.softmax"(%1357) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 2 : si32, numericStable = true}> : (tensor<1x511x128256xbf16, #ttnn_layout12>) -> tensor<1x511x128256xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%1357) <{force = false}> : (tensor<1x511x128256xbf16, #ttnn_layout12>) -> ()
        %1359 = "ttnn.to_layout"(%arg149) : (tensor<1x511x128256xbf16, #ttnn_layout8>) -> tensor<1x511x128256xbf16, #ttnn_layout12>
        "ttnn.deallocate"(%arg149) <{force = false}> : (tensor<1x511x128256xbf16, #ttnn_layout8>) -> ()
        %1360 = "ttnn.multiply"(%1358, %1359) : (tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x128256xbf16, #ttnn_layout12>) -> tensor<1x511x128256xbf16, #ttnn_layout12>
        %1361 = "ttnn.sum"(%1360) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x511x128256xbf16, #ttnn_layout12>) -> tensor<1x511x1xbf16, #ttnn_layout86>
        "ttnn.deallocate"(%1360) <{force = false}> : (tensor<1x511x128256xbf16, #ttnn_layout12>) -> ()
        %1362 = "ttnn.typecast"(%1361) : (tensor<1x511x1xbf16, #ttnn_layout86>) -> tensor<1x511x1xf32, #ttnn_layout60>
        "ttnn.deallocate"(%1361) <{force = false}> : (tensor<1x511x1xbf16, #ttnn_layout86>) -> ()
        %1363 = "ttnn.to_memory_config"(%1362) : (tensor<1x511x1xf32, #ttnn_layout60>) -> tensor<1x511x1xf32, #ttnn_layout13>
        %1364 = "ttnn.clamp_scalar"(%1362) <{max = 0x7F800000 : f32, min = 9.99999996E-13 : f32}> : (tensor<1x511x1xf32, #ttnn_layout60>) -> tensor<1x511x1xf32, #ttnn_layout60>
        "ttnn.deallocate"(%1362) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout60>) -> ()
        %1365 = "ttnn.to_memory_config"(%1364) : (tensor<1x511x1xf32, #ttnn_layout60>) -> tensor<1x511x1xf32, #ttnn_layout13>
        %1366 = "ttnn.log"(%1364) : (tensor<1x511x1xf32, #ttnn_layout60>) -> tensor<1x511x1xf32, #ttnn_layout60>
        "ttnn.deallocate"(%1364) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout60>) -> ()
        %1367 = "ttnn.to_layout"(%arg150) : (tensor<1x511x1xf32, #ttnn_layout9>) -> tensor<1x511x1xf32, #ttnn_layout13>
        "ttnn.deallocate"(%arg150) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout9>) -> ()
        %1368 = "ttnn.to_memory_config"(%1367) : (tensor<1x511x1xf32, #ttnn_layout13>) -> tensor<1x511x1xf32, #ttnn_layout60>
        %1369 = "ttnn.multiply"(%1366, %1368) : (tensor<1x511x1xf32, #ttnn_layout60>, tensor<1x511x1xf32, #ttnn_layout60>) -> tensor<1x511x1xf32, #ttnn_layout60>
        "ttnn.deallocate"(%1368) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout60>) -> ()
        "ttnn.deallocate"(%1366) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout60>) -> ()
        %1370 = "ttnn.sum"(%1369) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dim_arg = [0 : i32, 1 : i32, 2 : i32], keep_dim = true}> : (tensor<1x511x1xf32, #ttnn_layout60>) -> tensor<1x1x1xf32, #ttnn_layout87>
        "ttnn.deallocate"(%1369) <{force = false}> : (tensor<1x511x1xf32, #ttnn_layout60>) -> ()
        %1371 = "ttnn.to_memory_config"(%1370) : (tensor<1x1x1xf32, #ttnn_layout87>) -> tensor<1x1x1xf32, #ttnn_layout10>
        "ttnn.deallocate"(%1370) <{force = false}> : (tensor<1x1x1xf32, #ttnn_layout87>) -> ()
        %1372 = "ttnn.to_layout"(%arg147) : (tensor<1x512xsi32, #ttnn_layout7>) -> tensor<1x512xsi32, #ttnn_layout11>
        "ttnn.deallocate"(%arg147) <{force = false}> : (tensor<1x512xsi32, #ttnn_layout7>) -> ()
        return %1371, %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39, %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55, %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg75, %arg76, %arg77, %arg78, %arg79, %arg80, %arg81, %arg82, %arg83, %arg84, %arg85, %arg86, %arg87, %arg88, %arg89, %arg90, %arg91, %arg92, %arg93, %arg94, %arg95, %arg96, %arg97, %arg98, %arg99, %arg100, %arg101, %arg102, %arg103, %arg104, %arg105, %arg106, %arg107, %arg108, %arg109, %arg110, %arg111, %arg112, %arg113, %arg114, %arg115, %arg116, %arg117, %arg118, %arg119, %arg120, %arg121, %arg122, %arg123, %arg124, %arg125, %arg126, %arg127, %arg128, %arg129, %arg130, %arg131, %arg132, %arg133, %arg134, %arg135, %arg136, %arg137, %arg138, %arg139, %arg140, %arg141, %arg142, %arg143, %arg144, %arg145, %1372, %1359, %1367, %100, %109, %109, %112, %116, %123, %127, %147, %152, %160, %170, %170, %176, %180, %186, %186, %189, %192, %194, %196, %198, %200, %206, %211, %211, %214, %217, %123, %127, %233, %235, %240, %247, %247, %253, %257, %262, %262, %265, %267, %270, %272, %274, %276, %282, %287, %287, %290, %293, %123, %127, %309, %311, %316, %323, %323, %329, %333, %338, %338, %341, %343, %346, %348, %350, %352, %358, %363, %363, %366, %369, %123, %127, %385, %387, %392, %399, %399, %405, %409, %414, %414, %417, %419, %422, %424, %426, %428, %434, %439, %439, %442, %445, %123, %127, %461, %463, %468, %475, %475, %481, %485, %490, %490, %493, %495, %498, %500, %502, %504, %510, %515, %515, %518, %521, %123, %127, %537, %539, %544, %551, %551, %557, %561, %566, %566, %569, %571, %574, %576, %578, %580, %586, %591, %591, %594, %597, %123, %127, %613, %615, %620, %627, %627, %633, %637, %642, %642, %645, %647, %650, %652, %654, %656, %662, %667, %667, %670, %673, %123, %127, %689, %691, %696, %703, %703, %709, %713, %718, %718, %721, %723, %726, %728, %730, %732, %738, %743, %743, %746, %749, %123, %127, %765, %767, %772, %779, %779, %785, %789, %794, %794, %797, %799, %802, %804, %806, %808, %814, %819, %819, %822, %825, %123, %127, %841, %843, %848, %855, %855, %861, %865, %870, %870, %873, %875, %878, %880, %882, %884, %890, %895, %895, %898, %901, %123, %127, %917, %919, %924, %931, %931, %937, %941, %946, %946, %949, %951, %954, %956, %958, %960, %966, %971, %971, %974, %977, %123, %127, %993, %995, %1000, %1007, %1007, %1013, %1017, %1022, %1022, %1025, %1027, %1030, %1032, %1034, %1036, %1042, %1047, %1047, %1050, %1053, %123, %127, %1069, %1071, %1076, %1083, %1083, %1089, %1093, %1098, %1098, %1101, %1103, %1106, %1108, %1110, %1112, %1118, %1123, %1123, %1126, %1129, %123, %127, %1145, %1147, %1152, %1159, %1159, %1165, %1169, %1174, %1174, %1177, %1179, %1182, %1184, %1186, %1188, %1194, %1199, %1199, %1202, %1205, %123, %127, %1221, %1223, %1228, %1235, %1235, %1241, %1245, %1250, %1250, %1253, %1255, %1258, %1260, %1262, %1264, %1270, %1275, %1275, %1278, %1281, %123, %127, %1297, %1299, %1304, %1311, %1311, %1317, %1321, %1326, %1326, %1329, %1331, %1334, %1336, %1338, %1340, %1344, %1349, %1349, %1352, %1354, %1358, %1363, %1365 : tensor<1x1x1xf32, #ttnn_layout10>, tensor<128256x2048xbf16, #ttnn_layout>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<512x2048xbf16, #ttnn_layout2>, tensor<2048x2048xbf16, #ttnn_layout1>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<8192x2048xbf16, #ttnn_layout3>, tensor<2048x8192xbf16, #ttnn_layout4>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<2048xbf16, #ttnn_layout5>, tensor<1x512xsi32, #ttnn_layout11>, tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x1x512x64xbf16, #ttnn_layout16>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x512x64xf32, #ttnn_layout17>, tensor<1x32x64x512xf32, #ttnn_layout18>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x32x512x512xf32, #ttnn_layout19>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x8192xbf16, #ttnn_layout20>, tensor<1x512x2048xf32, #ttnn_layout14>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x1xf32, #ttnn_layout13>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x512x2048xbf16, #ttnn_layout15>, tensor<1x511x128256xbf16, #ttnn_layout12>, tensor<1x511x1xf32, #ttnn_layout13>, tensor<1x511x1xf32, #ttnn_layout13>
      }
    }
  }
}

