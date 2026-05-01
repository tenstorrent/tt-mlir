// RUN: ttmlir-opt --ttcore-register-device --d2m-split-unified-thread %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Test 1: Streaming remote_load with multicast parameters
  // Verifies: explicit CB form in DMA with mcast preserved, wait/pop in compute
  // CHECK-LABEL: func.func @test_streaming_remote_load
  func.func @test_streaming_remote_load(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // Datamovement: explicit CB form with multicast params
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}} mcore
    // CHECK-NOT: d2m.wait
    // CHECK-NOT: d2m.pop
    // CHECK-NOT: linalg.generic
    // Compute: wait + compute + pop
    // CHECK: }, {
    // CHECK: d2m.wait %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.core_index
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
          %2 = d2m.remote_load %cb_alloc %stream[%0, %1] mcore[%core0, %c0] mshape[%c1, %c4] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) outs(%cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %3 : !ttcore.tile<32x32, f32>
          }
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    memref.dealloc %cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }

  // Test 2: Streaming remote_store
  // Verifies: explicit CB form in DMA, reserve/push in compute, external alloc preserved
  // CHECK-LABEL: func.func @test_streaming_remote_store
  func.func @test_streaming_remote_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                                          %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 1 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // Datamovement: explicit CB form store
    // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %{{.*}}
    // CHECK-NOT: d2m.reserve
    // CHECK-NOT: d2m.push
    // Compute: reserve + push
    // CHECK: }, {
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: d2m.push %{{.*}}
    // CHECK-NOT: d2m.remote_store
    // CHECK-NOT: d2m.core_index
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        additionalArgs(%cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          %2 = d2m.remote_store %stream[%0, %1] %cb_alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    memref.dealloc %cb_alloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }

  // Test 3: Full streaming load + store
  // Verifies: both ops in DMA, all CB sync ops in compute with linalg.generic
  // CHECK-LABEL: func.func @test_full_streaming_load_store
  func.func @test_full_streaming_load_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                            %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream_in = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_in = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // Datamovement: both in explicit CB form
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %{{.*}}
    // CHECK-NOT: d2m.wait
    // CHECK-NOT: d2m.reserve
    // CHECK-NOT: d2m.push
    // CHECK-NOT: d2m.pop
    // CHECK-NOT: linalg.generic
    // Compute: reserve at top, then wait/compute/push/pop inside loops
    // CHECK: }, {
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: d2m.wait %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.push %{{.*}}
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.remote_store
    // CHECK-NOT: d2m.core_index
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        additionalArgs(%cb_in : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          %2 = d2m.remote_load %cb_in %stream_in[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
          %alloc_out = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb_in : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) outs(%alloc_out : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %3 : !ttcore.tile<32x32, f32>
          }
          %4 = d2m.remote_store %stream_out[%0, %1] %alloc_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    memref.dealloc %cb_in : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }

  // Test 4: Aliased load/store (all L1, no DMA needed)
  // Verifies: no remote ops in DMA, full CB sync pattern in compute with
  // linalg.generic and pop after last use
  // CHECK-LABEL: func.func @test_aliased_load_store
  func.func @test_aliased_load_store() {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA: no remote ops (aliased)
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.remote_store
    // Compute: full CB ops with pop after linalg
    // CHECK: }, {
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: d2m.push %{{.*}}
    // CHECK: d2m.wait %{{.*}}
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: d2m.push %{{.*}}
    // CHECK: d2m.wait %{{.*}}
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.tile_add
    // CHECK: d2m.pop %{{.*}}
    // CHECK: d2m.pop %{{.*}}
    // CHECK: d2m.push %{{.*}}
    // CHECK: d2m.wait %{{.*}}
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.remote_store
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x3>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc, %alloc_0 : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%alloc_1 : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)  {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          %buffer0 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
          %buffer1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
          %buffer = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
          %2 = d2m.remote_load %buffer0 %alloc[%0, %1] : memref<1x1x!ttcore.tile<32x32, f32>>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>>
          %5 = d2m.remote_load %buffer1 %alloc_0[%0, %1] : memref<1x1x!ttcore.tile<32x32, f32>>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>>

          linalg.generic {
            indexing_maps = [#map, #map, #map],
            iterator_types = ["parallel", "parallel"]
          } ins(%2, %5 : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>)
            outs(%buffer : memref<1x1x!ttcore.tile<32x32, f32>>) {
          ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %add = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %add : !ttcore.tile<32x32, f32>
          }

          %result = d2m.remote_store %alloc_1[%0, %1] %buffer : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>> -> memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 5: Non-unified thread (should not be transformed)
  // CHECK-LABEL: func.func @test_non_unified_no_change
  func.func @test_non_unified_no_change(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<compute>]
    // CHECK-NOT: #d2m.thread<datamovement>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^compute0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 6: Multiple streaming loads from different operands
  // Verifies: each load gets its own CB in both threads
  // CHECK-LABEL: func.func @test_multiple_remote_loads
  func.func @test_multiple_remote_loads(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                         %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %cb1 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64, d2m.cb_for_operand = 1 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA: two loads in explicit CB form
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // CHECK-NOT: d2m.wait
    // Compute: two waits, compute, two pops
    // CHECK: }, {
    // CHECK: d2m.wait %{{.*}}
    // CHECK: d2m.wait %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.pop %{{.*}}
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%cb0, %cb1 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %2 = d2m.remote_load %cb0 %stream0[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      %3 = d2m.remote_load %cb1 %stream1[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%cb0, %cb1 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>)
        outs(%cb0 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
      ^bb0(%in0: !ttcore.tile<32x32, f32>, %in1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %add = "d2m.tile_add"(%in0, %in1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %add : !ttcore.tile<32x32, f32>
      }
    }
    memref.dealloc %cb0 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %cb1 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }

  // Test 7: Shared buffer pair (both remote, DMA-only)
  // Verifies: both ops share one CB (output operand's CB), compute empty
  // CHECK-LABEL: func.func @test_shared_buffer_pair
  func.func @test_shared_buffer_pair(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream_in = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA: both ops use the same shared CB (output operand's CB)
    // CHECK: %[[SHARED_CB:.*]] = d2m.get_cb(1) operand_index = 1
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[SHARED_CB]]
    // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[SHARED_CB]]
    // CHECK: }, {
    // Compute: nothing (DMA-only)
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.remote_store
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %shared = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %shared %stream_in[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_store %stream_out[%core0, %core1] %shared : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

  // Test 8: Shared buffer pair — load remote, store local
  // Verifies: DMA keeps load, compute gets wait+pop for aliased store
  // CHECK-LABEL: func.func @test_shared_pair_load_remote_store_local
  func.func @test_shared_pair_load_remote_store_local(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %stream_in = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA: only the remote load
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // Compute: wait+pop
    // CHECK: }, {
    // CHECK: d2m.wait %{{.*}}
    // CHECK: d2m.pop %{{.*}}
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %shared = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %shared %stream_in[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_store %arg1[%core0, %core1] %shared : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

  // Test 9: Shared buffer pair — load local, store remote
  // Verifies: DMA store uses input (aliased load's) CB, compute reserve+push
  // uses the same CB so both threads agree on the port.
  // CHECK-LABEL: func.func @test_shared_pair_load_local_store_remote
  func.func @test_shared_pair_load_local_store_remote(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA: store uses input (aliased) operand's CB
    // CHECK: %[[INPUT_CB:.*]] = d2m.get_cb(0) operand_index = 0
    // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[INPUT_CB]]
    // Compute: reserve+push using the same input CB
    // CHECK: }, {
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: d2m.push %{{.*}}
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %shared = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %shared %arg0[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_store %stream_out[%core0, %core1] %shared : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

  // Test 10: L1-to-L1 shared buffer copy (both operands are L1 shards)
  // CHECK-LABEL: func.func @test_l1_to_l1_shared_buffer_copy
  func.func @test_l1_to_l1_shared_buffer_copy() {
    %input = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %output = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %{{.*}}
    // Compute is emtpy
    // CHECK: }, {
    // CHECK-NEXT: }
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x3>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%input : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%output : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %buf = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %buf %input[%core0, %core1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_store %output[%core0, %core1] %buf : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

  // Test 11: Self-read/write (read-modify-write on same buffer)
  // Verifies: no remote ops in DMA, compute has reserve+push+wait for the
  // accumulator CB, no extra push/wait/pop from store
  // CHECK-LABEL: func.func @test_self_read_write_accumulator
  func.func @test_self_read_write_accumulator() {
    %output = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %input_view = d2m.view_layout %output remapping = #map4 : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %cb_buf = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]

    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // CHECK-NOT: d2m.remote_store

    // CHECK: }, {
    // CHECK: scf.for
    // CHECK: d2m.wait
    // CHECK: d2m.reserve
    // CHECK: d2m.push
    // CHECK: d2m.wait
    // CHECK: d2m.tile_add
    // CHECK: d2m.pop
    // CHECK: d2m.pop
    // CHECK-NOT: memref.alloc
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.remote_store
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x3>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%input_view : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%output : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%cb_buf : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        %0 = d2m.remote_load %cb_buf %input_view[%core0, %arg0] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
        %acc_buf = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %1 = d2m.remote_load %acc_buf %output[%core0, %core1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        // Compute: add input tile to accumulator
        %in_tile = memref.load %cb_buf[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
        %acc_tile = memref.load %acc_buf[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        %sum = "d2m.tile_add"(%in_tile, %acc_tile) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        memref.store %sum, %acc_buf[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        // Self-write: store accumulator back to the same output
        %2 = d2m.remote_store %output[%core0, %core1] %acc_buf : memref<4x3x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      } {d2m.blocking_loop = 1}
    }
    memref.dealloc %cb_buf : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    return
  }

  // Test 12: Streaming remote_load + implicit local_copy (DM-only src consumer)
  // Verifies: when source buffer is consumed ONLY by local_copy (no compute
  // consumer), the pop for the source CB is deferred to the DM thread.
  // CHECK-LABEL: func.func @test_local_copy_implicit_dm_only_src
  func.func @test_local_copy_implicit_dm_only_src(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_buf = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %scratch_buf = memref.alloc() {address = 9216 : i64, alignment = 16 : i64, d2m.cb_for_operand = 3 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA thread: remote_load into CB, wait, local_copy into scratch CB
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // CHECK: d2m.wait
    // CHECK: d2m.local_copy %{{.*}} into %{{.*}} indexing_maps
    // CHECK-NOT: linalg.generic

    // Compute thread: only wait on scratch CB (source pop is deferred)
    // CHECK: }, {
    // CHECK: d2m.wait
    // CHECK: linalg.generic
    // CHECK: d2m.pop
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.local_copy
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%cb_buf, %scratch_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      // Streaming load from DRAM into the CB buffer
      %0 = d2m.remote_load %cb_buf %stream[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      // Local copy rearranges data into scratch buffer
      d2m.local_copy %cb_buf, %scratch_buf indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      // Compute reads ONLY from scratch (not from cb_buf)
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%scratch_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>)
        outs(%scratch_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %2 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %2 : !ttcore.tile<32x32, f32>
      }
    }
    memref.dealloc %cb_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %scratch_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }

  // Test 13: copy -> copy (chained local_copies, cb0 -> cb1 -> cb2)
  // Verifies: DMA chains waits through intermediate CBs. Compute only
  // waits on the final CB. Intermediate CBs stay entirely on DMA.
  // CHECK-LABEL: func.func @test_copy_chain
  func.func @test_copy_chain(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_buf = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %scratch1 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64, d2m.cb_for_operand = 3 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %scratch2 = memref.alloc() {address = 13312 : i64, alignment = 16 : i64, d2m.cb_for_operand = 4 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA: load into cb0, wait cb0, copy into cb1, wait cb1, copy into cb2
    // CHECK: d2m.remote_load %{{.*}} into %{{.*}}
    // CHECK: d2m.wait
    // CHECK: d2m.local_copy %{{.*}} into %{{.*}}
    // CHECK: d2m.wait
    // CHECK: d2m.local_copy %{{.*}} into %{{.*}}
    // CHECK-NOT: linalg.generic

    // Compute: only wait on final cb2, compute, pop cb2
    // CHECK: }, {
    // CHECK: d2m.wait
    // CHECK: linalg.generic
    // CHECK: d2m.pop
    // CHECK-NOT: d2m.local_copy
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%cb_buf, %scratch1, %scratch2 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %0 = d2m.remote_load %cb_buf %stream[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      d2m.local_copy %cb_buf, %scratch1 indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      d2m.local_copy %scratch1, %scratch2 indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%scratch2 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>)
        outs(%scratch2 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %2 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %2 : !ttcore.tile<32x32, f32>
      }
    }
    memref.dealloc %cb_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %scratch1 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %scratch2 : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }

  // Test 14: copy -> remote_store (local_copy feeds a remote_store, DMA-only)
  // Verifies: both local_copy and remote_store go to DMA. Compute is empty.
  // CHECK-LABEL: func.func @test_copy_to_store
  func.func @test_copy_to_store(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream_in = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_buf = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %scratch_buf = memref.alloc() {address = 9216 : i64, alignment = 16 : i64, d2m.cb_for_operand = 2 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA: load, wait, copy, store all on DMA
    // CHECK: d2m.remote_load %{{.*}} into %{{.*}}
    // CHECK: d2m.wait
    // CHECK: d2m.local_copy %{{.*}} into %{{.*}}
    // CHECK: d2m.remote_store %{{.*}} from %{{.*}}
    // CHECK-NOT: linalg.generic

    // Compute: empty (DMA-only pipeline)
    // CHECK: }, {
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.local_copy
    // CHECK-NOT: d2m.remote_store
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        additionalArgs(%cb_buf, %scratch_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %0 = d2m.remote_load %cb_buf %stream_in[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      d2m.local_copy %cb_buf, %scratch_buf indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      %1 = d2m.remote_store %stream_out[%core0, %core1] %scratch_buf : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    }
    memref.dealloc %cb_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %scratch_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }

  // Test 15: compute -> copy -> compute
  // Verifies: compute gets reserve + push for the source CB (compute-produced).
  // DMA gets wait on source CB + local_copy into destination CB.
  // Second compute step reads from the copy destination.
  // CHECK-LABEL: func.func @test_compute_to_copy
  func.func @test_compute_to_copy(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %cb_buf = memref.alloc() {address = 5120 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    // Compute writes into this scratch CB, local_copy reads from it
    %compute_out_buf = memref.alloc() {address = 9216 : i64, alignment = 16 : i64, d2m.cb_for_operand = 3 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    // local_copy writes into this scratch CB, second compute reads from it
    %copy_dst_buf = memref.alloc() {address = 13312 : i64, alignment = 16 : i64, d2m.cb_for_operand = 4 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // DMA: load, wait on compute's push for src CB, copy into dst CB
    // CHECK: d2m.remote_load %{{.*}} into %{{.*}}
    // CHECK: d2m.wait
    // CHECK: d2m.local_copy %{{.*}} into %{{.*}}
    // CHECK-NOT: linalg.generic
    // CHECK-NOT: d2m.reserve
    // CHECK-NOT: d2m.push

    // Compute: reserve compute scratch (at alloc position), wait load CB,
    //          first compute, push scratch, wait copy dst CB, second compute,
    //          pop dst CB, pop load CB
    // CHECK: }, {
    // CHECK: d2m.wait
    // CHECK: linalg.generic
    // CHECK: d2m.push
    // CHECK: d2m.wait
    // CHECK: linalg.generic
    // CHECK: d2m.pop
    // CHECK: d2m.pop
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.local_copy
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%cb_buf, %compute_out_buf, %copy_dst_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      // Load input from DRAM
      %0 = d2m.remote_load %cb_buf %stream[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      // First compute: reads from load CB, writes to compute_out scratch
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%cb_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>)
        outs(%compute_out_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %2 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %2 : !ttcore.tile<32x32, f32>
      }
      // local_copy rearranges compute output into copy_dst
      d2m.local_copy %compute_out_buf, %copy_dst_buf indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      // Second compute: reads from copy_dst
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%copy_dst_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>)
        outs(%copy_dst_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %2 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %2 : !ttcore.tile<32x32, f32>
      }
    }
    memref.dealloc %cb_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %compute_out_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    memref.dealloc %copy_dst_buf : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    return
  }
}
