// RUN: ttmlir-opt --canonicalize %s | FileCheck %s

// Regression test: block_offset must survive canonicalize.
// CHECK-COUNT-8: d2m.block_offset(

module attributes {ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @interleaved_dma(%arg0: memref<64x128xf32>) -> memref<64x128xf32> attributes {tt.function_type = "forward_device"} {
    %alloc = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_0 = memref.alloc() : memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #ttcore.memory_space<l1>>
    d2m.to_device %arg0, %alloc_0 layout = <logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)> : memref<64x128xf32> into memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%alloc_0 : memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #ttcore.memory_space<l1>>)
        outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<l1>>) {
    ^unified0(%cb0: !d2m.cb<memref<64x128xf32, #ttcore.memory_space<l1>>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>):
      %block_offset1 = d2m.block_offset(1) : index
      %block_offset0 = d2m.block_offset(0) : index
      %block_factor0 = d2m.get_block_factor(0) : index
      %block_factor1 = d2m.get_block_factor(1) : index
      affine.for %arg1 = 0 to %block_factor0 {
        affine.for %arg2 = 0 to %block_factor1 {
          %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%block_offset0]
          %1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg2)[%block_offset1]
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32>
          %2 = d2m.remote_load %alloc_6 %alloc_0[%0, %1] : memref<64x128xf32>, memref<1x1x64x128xf32, #ttcore.shard<512x4, 1>, #ttcore.memory_space<l1>> -> memref<64x128xf32, #ttcore.memory_space<l1>>
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
          %3 = "d2m.tile_tilize_block"(%alloc_6, %alloc_7) : (memref<64x128xf32>, memref<2x4x!ttcore.tile<32x32, f32>>) -> memref<2x4x!ttcore.tile<32x32, f32>>
          %4 = d2m.remote_store %alloc[%0, %1] %alloc_7 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<l1>>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<l1>>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
    %alloc_1 = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #ttcore.memory_space<dram>>
    %view = d2m.view_layout %alloc_1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #ttcore.memory_space<dram>> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<dram>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<l1>>)
        outs(%view : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<dram>>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>):
      %block_offset1 = d2m.block_offset(1) : index
      %block_offset0 = d2m.block_offset(0) : index
      %block_factor0 = d2m.get_block_factor(0) : index
      %block_factor1 = d2m.get_block_factor(1) : index
      affine.for %arg1 = 0 to %block_factor0 {
        affine.for %arg2 = 0 to %block_factor1 {
          %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%block_offset0]
          %1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg2)[%block_offset1]
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
          %2 = d2m.remote_load %alloc_6 %alloc[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.memory_space<l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
          %3 = d2m.remote_store %view[%0, %1] %alloc_6 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #ttcore.memory_space<dram>>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #ttcore.memory_space<dram>>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
    %alloc_2 = memref.alloc() : memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>>
    %view_3 = d2m.view_layout %alloc_1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #ttcore.memory_space<dram>> -> memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (0, 0, (d0 + (d1 * 2 + d3) floordiv 4) mod 2, (d1 * 2 + d3) mod 4)>, #ttcore.memory_space<dram>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%view_3 : memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (0, 0, (d0 + (d1 * 2 + d3) floordiv 4) mod 2, (d1 * 2 + d3) mod 4)>, #ttcore.memory_space<dram>>)
        outs(%alloc_2 : memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>>) {
    ^unified0(%cb0: !d2m.cb<memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb1: !d2m.cb<memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>):
      %block_offset1 = d2m.block_offset(1) : index
      %block_offset0 = d2m.block_offset(0) : index
      %block_factor0 = d2m.get_block_factor(0) : index
      %block_factor1 = d2m.get_block_factor(1) : index
      affine.for %arg1 = 0 to %block_factor0 {
        affine.for %arg2 = 0 to %block_factor1 {
          %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%block_offset0]
          %1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg2)[%block_offset1]
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x2x!ttcore.tile<32x32, f32>>
          %2 = d2m.remote_load %alloc_6 %view_3[%0, %1] : memref<1x2x!ttcore.tile<32x32, f32>>, memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (0, 0, (d0 + (d1 * 2 + d3) floordiv 4) mod 2, (d1 * 2 + d3) mod 4)>, #ttcore.memory_space<dram>> -> memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
          %3 = d2m.remote_store %alloc_2[%0, %1] %alloc_6 : memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>>, memref<1x2x!ttcore.tile<32x32, f32>> -> memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
    %alloc_4 = memref.alloc() : memref<64x128xf32>
    %alloc_5 = memref.alloc() : memref<2x2x32x64xf32, #ttcore.shard<256x4, 1>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%alloc_2 : memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>>)
        outs(%alloc_5 : memref<2x2x32x64xf32, #ttcore.shard<256x4, 1>, #ttcore.memory_space<l1>>) {
    ^unified0(%cb0: !d2m.cb<memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %cb1: !d2m.cb<memref<32x64xf32, #ttcore.memory_space<l1>>>):
      %block_offset1 = d2m.block_offset(1) : index
      %block_offset0 = d2m.block_offset(0) : index
      %block_factor0 = d2m.get_block_factor(0) : index
      %block_factor1 = d2m.get_block_factor(1) : index
      affine.for %arg1 = 0 to %block_factor0 {
        affine.for %arg2 = 0 to %block_factor1 {
          %0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg1)[%block_offset0]
          %1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%arg2)[%block_offset1]
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x2x!ttcore.tile<32x32, f32>>
          %2 = d2m.remote_load %alloc_6 %alloc_2[%0, %1] : memref<1x2x!ttcore.tile<32x32, f32>>, memref<2x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
          %3 = "d2m.tile_untilize_block"(%alloc_6, %alloc_7) : (memref<1x2x!ttcore.tile<32x32, f32>>, memref<32x64xf32>) -> memref<32x64xf32>
          %4 = d2m.remote_store %alloc_5[%0, %1] %alloc_7 : memref<2x2x32x64xf32, #ttcore.shard<256x4, 1>, #ttcore.memory_space<l1>>, memref<32x64xf32> -> memref<2x2x32x64xf32, #ttcore.shard<256x4, 1>, #ttcore.memory_space<l1>>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
    d2m.to_host %alloc_5, %alloc_4 layout = <logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)> : memref<2x2x32x64xf32, #ttcore.shard<256x4, 1>, #ttcore.memory_space<l1>> into memref<64x128xf32>
    return %alloc_4 : memref<64x128xf32>
  }
}
