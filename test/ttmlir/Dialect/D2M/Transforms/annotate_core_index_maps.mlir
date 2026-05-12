// RUN: ttmlir-opt --ttcore-register-device --d2m-annotate-core-index-maps %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#grid_v2p = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @annotate_core_index_maps
  func.func @annotate_core_index_maps(%arg0: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
    // CHECK: d2m.core_index(0) {phys_to_virt_map = {{.*}}} : index
    // CHECK: d2m.core_index(1) {phys_to_virt_map = {{.*}}} : index
    d2m.generic {block_factors = [], grid = #grid_v2p, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
    }
    return
  }
}
