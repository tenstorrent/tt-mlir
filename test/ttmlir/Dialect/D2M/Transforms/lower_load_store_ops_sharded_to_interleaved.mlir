// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-load-store-ops-to-explicit-cb-form %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 mod 64)>
#phys_to_virt = affine_map<(d0, d1) -> (0, 0, (d1 + d0 * 8) mod 64)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @sharded_to_interleaved_writeback
  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x64, (d0, d1) -> (0, 0, (d1 + d0 * 8) mod 64)>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: %[[CB:.*]] = d2m.get_cb(0) operand_index = 0
  // CHECK: d2m.reserve %[[CB]]
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[CB]]
  // CHECK: d2m.push %[[CB]]
  func.func @sharded_to_interleaved_writeback(
      %arg0: memref<1x64x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>) {
    %dram_alloc = memref.alloc() {address = 1024 : i64, alignment = 32 : i64} : memref<1x1x1x64x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<131072x2048>, #dram>
    %view = d2m.view_layout %dram_alloc remapping = #map : memref<1x1x1x64x!ttcore.tile<32x32, bf16>, #ttcore.interleaved<131072x2048>, #dram> -> memref<1x64x1x1x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x64, (d0, d1) -> (0, 0, (d1 + d0 * 8) mod 64)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<1x64x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1>)
        outs(%view : memref<1x64x1x1x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #dram>)
     {
    ^unified0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, bf16>, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = #phys_to_virt} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = #phys_to_virt} : index
      %alloc_local = memref.alloc() {alignment = 64 : i64} : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %0 = d2m.remote_load %alloc_local %arg0[%core0, %core1] : memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, memref<1x64x1x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
      %1 = d2m.remote_store %view[%core0, %core1] %alloc_local : memref<1x64x1x1x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #dram>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1> -> memref<1x1x!ttcore.tile<32x32, bf16>, #l1>
    }
    return
  }
}
