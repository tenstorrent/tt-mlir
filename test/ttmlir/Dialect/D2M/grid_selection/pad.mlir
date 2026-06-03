// RUN: ttmlir-opt --d2m-grid-selection %s | FileCheck %s

// Input is post-MaterializeViewReturns IR: a DM-only d2m.generic consumes a
// composite_view whose inputs are a to_layout (real data) and a fill_buffer
// (pad value).
//
// GridSelection picks 7x8 for the DM generic (output 224x256 → 7x8 tiles) and
// reblocks composite_view + its inputs. The fill_buffer takes Path Z reblock:
// consumer_grid (7x8) × fixed_shard (32x32) → per-core lock-step stamp.

#layout = #ttcore.metal_layout<logical_shape = 224x224, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#layout1 = #ttcore.metal_layout<logical_shape = 224x32, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#layout2 = #ttcore.metal_layout<logical_shape = 224x256, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @pad_only
  func.func @pad_only(%arg0: tensor<224x224xbf16>) -> tensor<224x256xbf16> {
    %0 = d2m.empty() : tensor<1x1x224x224xbf16, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<224x224xbf16> into tensor<1x1x224x224xbf16, #layout> -> tensor<1x1x224x224xbf16, #layout>
    // fill_buffer reblocked to consumer 7x8 grid × fixed_shard 32x32.
    // CHECK: d2m.fill_buffer
    // CHECK-SAME: fixed_shard = array<i64: 32, 32>
    // CHECK-SAME: value = 1.000000e+00 : bf16
    // CHECK-SAME: tensor<7x8x32x32xbf16
    %2 = d2m.fill_buffer() {fixed_shard = array<i64: 32, 32>, value = 1.000000e+00 : bf16} : tensor<1x1x224x32xbf16, #layout1>
    // composite_view rebuilt at 7x8 grid (output side).
    // CHECK: d2m.composite_view
    // CHECK-SAME: dim = 1 : si32
    // CHECK-SAME: logicalSizes = array<i64: 224, 32>
    // CHECK-SAME: tensor<7x8x32x32xbf16
    %3 = "d2m.composite_view"(%1, %2) <{dim = 1 : si32, logicalSizes = array<i64: 224, 32>}> : (tensor<1x1x224x224xbf16, #layout>, tensor<1x1x224x32xbf16, #layout1>) -> tensor<1x1x224x256xbf16, #layout2>
    %4 = d2m.empty() : tensor<224x256xbf16>
    %5 = d2m.empty() : tensor<1x1x224x256xbf16, #layout2>
    // DM-only generic running on the picked 7x8 grid.
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<7x8>
    %6 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%3 : tensor<1x1x224x256xbf16, #layout2>)
        outs(%5 : tensor<1x1x224x256xbf16, #layout2>)
     {
      %8 = tensor.empty() : tensor<224x256xbf16>
      %block0 = d2m.block_index(0) : index
      %block1 = d2m.block_index(1) : index
      %9 = d2m.remote_load %8 %3[%block0, %block1] : tensor<224x256xbf16>, tensor<1x1x224x256xbf16, #layout2> -> tensor<224x256xbf16>
      %10 = d2m.remote_store %5[%block0, %block1] %9 : tensor<1x1x224x256xbf16, #layout2>, tensor<224x256xbf16> -> tensor<1x1x224x256xbf16, #layout2>
      d2m.yield %10 : (tensor<1x1x224x256xbf16, #layout2>)
    } : tensor<1x1x224x256xbf16, #layout2>
    %7 = d2m.to_layout %6, %4 : tensor<1x1x224x256xbf16, #layout2> into tensor<224x256xbf16> -> tensor<224x256xbf16>
    return %7 : tensor<224x256xbf16>
  }
}
