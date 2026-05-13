// RUN: ttmlir-opt --ttcore-register-device --d2m-preallocate-mcast-semaphores --d2m-lower-load-store-ops-to-dma --d2m-normalize-thread-args --d2m-generic-regions-to-funcs %s | FileCheck %s

// Exercise multicast d2m.remote_load lowering when the enclosing d2m.generic
// uses a virtualized grid. Generic region outlining materializes multicast
// DMA/semaphore core operands in physical grid space before kernel lowering.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#grid_v2p = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  func.func @mcast_remote_load_core_index_mapping(%arg0: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
    // CHECK-LABEL: func.func private @datamovement_kernel0
    // CHECK: %[[CORE0:.*]] = d2m.core_index(0) {phys_to_virt_map = {{.*}}} : index
    // CHECK: %[[DMA_CORE0:.*]] = affine.apply {{.*}}(%[[CORE0]], %c0)
    // CHECK: %[[DMA_CORE1:.*]] = affine.apply {{.*}}(%[[CORE0]], %c0)
    // CHECK: d2m.dma_write %{{.*}}, %{{.*}} core[%[[DMA_CORE0]], %[[DMA_CORE1]]] mcast[%c1, %c2]
    // CHECK: %[[SET_CORE0:.*]] = affine.apply {{.*}}(%[[CORE0]], %c0)
    // CHECK: %[[SET_CORE1:.*]] = affine.apply {{.*}}(%[[CORE0]], %c0)
    // CHECK: d2m.semaphore_set %{{.*}}, %c1, core[%[[SET_CORE0]], %[[SET_CORE1]]] mcast[%c1, %c2]
    // CHECK: %[[INC_CORE0:.*]] = affine.apply {{.*}}(%[[CORE0]], %c0)
    // CHECK: %[[INC_CORE1:.*]] = affine.apply {{.*}}(%[[CORE0]], %c0)
    // CHECK: d2m.semaphore_inc %{{.*}}, %c1, core[%[[INC_CORE0]], %[[INC_CORE1]]]
    d2m.generic {block_factors = [1, 1], grid = #grid_v2p, indexing_maps = [#map, #map], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>)  {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
          d2m.remote_load %arg0[%0, %1] into %cb0 mcore[%core0, %c0] mshape[%c1, %c2] : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
}
