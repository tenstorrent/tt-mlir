// RUN: ttmlir-opt --ttcore-register-device --d2m-split-unified-thread %s | FileCheck %s

// Fan-out: a CB streamed in once but consumed by two compute ops in the same
// block (as softmax reuses `scores` for both the max-reduction and the
// subtract). The producer pushes the CB once per blocking-loop iteration, so
// the compute side must emit exactly one wait/pop pair wrapping both
// consumers -- not one wait/pop per consumer, which would starve the second
// consumer of a depth-limited CB and deadlock on device.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @test_fanout_shared_input
  func.func @test_fanout_shared_input(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %res = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_in = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %cb_tmp = memref.alloc() {address = 6144 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %cb_out = memref.alloc() {address = 7168 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // Compute thread: the shared input CB (%cb_in) is consumed by both compute
    // ops, so it gets a single wait before the first consumer and a single pop
    // after the last -- not one wait/pop per consumer.
    // CHECK: }, {
    // CHECK: d2m.wait %[[IN:[0-9a-z_]+]]
    // CHECK-NOT: d2m.wait %[[IN]]
    // CHECK: d2m.tile_exp
    // CHECK-NOT: d2m.wait %[[IN]]
    // CHECK: d2m.tile_add
    // CHECK: d2m.pop %[[IN]]
    // CHECK-NOT: d2m.pop %[[IN]]
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%res : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%cb_in, %cb_tmp, %cb_out : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
          d2m.remote_load %cb_in %stream[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb_in : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) outs(%cb_tmp : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %3 : !ttcore.tile<32x32, f32>
          }
          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb_in, %cb_tmp : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) outs(%cb_out : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
          ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %3 : !ttcore.tile<32x32, f32>
          }
          d2m.remote_store %res[%0, %1] %cb_out : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    memref.dealloc %cb_in : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %cb_tmp : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %cb_out : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }
}
