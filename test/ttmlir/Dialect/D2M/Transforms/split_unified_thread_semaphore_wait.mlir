// RUN: ttmlir-opt --ttcore-register-device --d2m-split-unified-thread %s 2>&1 | FileCheck %s

// This test file verifies that d2m.semaphore_wait ops (without reset values) are
// properly replicated when splitting a unified thread into datamovement and compute
// regions.
//
// IMPORTANT: d2m.semaphore_wait with reset values are NOT supported in unified thread
// form. The pass will emit an error if a semaphore_wait with reset is encountered,
// because replicating the reset across both threads would break synchronization.
// Use separate d2m.semaphore_set ops if reset functionality is needed.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Test 1: semaphore_wait alongside streaming remote_load (implicit form)
  // Verifies:
  // - semaphore_wait is replicated into both datamovement and compute regions
  // - remote_load converted to explicit CB form in datamovement only
  // - compute gets wait/pop from splitter + semaphore_wait
  // - relative order of semaphore_wait is preserved in each region
  // CHECK-LABEL: func.func @test_semaphore_wait_with_remote_load
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  func.func @test_semaphore_wait_with_remote_load(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // Datamovement region: remote_load in explicit CB form + semaphore_wait
    // CHECK: ^datamovement0
    // CHECK: scf.for
    // CHECK: d2m.core_index(0)
    // CHECK: d2m.core_index(1)
    // CHECK: scf.for
    // CHECK: arith.addi
    // CHECK: arith.addi
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // CHECK: d2m.semaphore_wait %{{.*}}, %{{.*}}
    // CHECK-NOT: d2m.wait
    // CHECK-NOT: linalg.generic
    // CHECK-NOT: d2m.pop

    // Compute region: wait + semaphore_wait + compute + pop
    // CHECK: ^compute0
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.wait %{{.*}}
    // CHECK: d2m.semaphore_wait %{{.*}}, %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.core_index
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0(%sem0: !d2m.local_semaphore):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
          %2 = d2m.remote_load %cb_alloc %stream[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
          d2m.semaphore_wait %sem0, %c1 : !d2m.local_semaphore
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) outs(%cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %3 : !ttcore.tile<32x32, f32>
          }
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    memref.dealloc %cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }

  // Test 2: semaphore_wait with no remote ops (aliased operands)
  // Verifies:
  // - semaphore_wait is replicated into both regions even with no remote ops
  // - aliased load converted to CB ops in compute
  // - DMA region has semaphore_wait but no remote ops
  // CHECK-LABEL: func.func @test_semaphore_wait_no_remote
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  func.func @test_semaphore_wait_no_remote(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // Datamovement region: semaphore_wait replicated, no remote ops
    // CHECK: ^datamovement0
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.semaphore_wait %{{.*}}, %{{.*}}
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.wait
    // CHECK-NOT: linalg.generic
    // CHECK-NOT: d2m.pop

    // Compute region: semaphore_wait + aliased CB ops
    // CHECK: ^compute0
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.semaphore_wait %{{.*}}, %{{.*}}
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: d2m.push %{{.*}}
    // CHECK: d2m.wait %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0(%sem0: !d2m.local_semaphore):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          d2m.semaphore_wait %sem0, %c1 : !d2m.local_semaphore
          %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>>
          %0 = d2m.remote_load %buffer %arg0[%c0, %c0] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<2x4x!ttcore.tile<32x32, f32>>) outs(%0 : memref<2x4x!ttcore.tile<32x32, f32>>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %1 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %1 : !ttcore.tile<32x32, f32>
          }
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }
}
