// RUN: ttmlir-opt --ttcore-register-device --d2m-split-unified-thread --split-input-file --verify-diagnostics %s

// Cross-nest fan-out: a CB consumed by compute ops living in two distinct loop
// nests. The producer pushes the CB once per blocking-loop iteration, so a
// single per-block wait/pop pair cannot dominate/post-dominate consumers spread
// across separate nests. This is not yet supported and must be diagnosed (not
// silently miscompiled into a deadlock).

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>], supported_tile_sizes = [ 32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(0,0)], dram_bank_to_logical_worker_noc1 = [(0,0)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, 0, 0), meshShape = , chipIds = [0]>
  func.func @fanout_cross_nest(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %res = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_in = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %cb_a = memref.alloc() {address = 6144 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    // expected-error @+1 {{compute ops span multiple synchronization scopes (e.g. a CB consumed across distinct loop nests); cross-nest fan-out is not yet supported}}
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%res : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%cb_in, %cb_a : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      d2m.remote_load %cb_in %stream[%c0, %c0] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
      scf.for %i = %c0 to %c1 step %c1 {
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb_in : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) outs(%cb_a : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
        ^bb0(%in: !ttcore.tile<32x32, f32>, %o: !ttcore.tile<32x32, f32>):
          %e = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %e : !ttcore.tile<32x32, f32>
        }
      } {d2m.blocking_loop = 0}
      scf.for %j = %c0 to %c1 step %c1 {
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb_in : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) outs(%cb_a : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
        ^bb0(%in: !ttcore.tile<32x32, f32>, %o: !ttcore.tile<32x32, f32>):
          %e = "d2m.tile_log"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          linalg.yield %e : !ttcore.tile<32x32, f32>
        }
      } {d2m.blocking_loop = 1}
      d2m.remote_store %res[%c0, %c0] %cb_a : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    }
    return
  }
}
