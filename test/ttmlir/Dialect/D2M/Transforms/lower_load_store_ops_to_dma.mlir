// UNSUPPORTED: true
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-load-store-ops --d2m-generate-outer-loops --d2m-split-unified-thread --d2m-lower-load-store-ops-to-dma -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  // Test 1: Simple case with remote_load - should be lowered to dma_read in datamovement region
  // CHECK-LABEL: func.func @test_simple_remote_load
  func.func @test_simple_remote_load(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: ^datamovement0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     d2m.core_index(0)
      // CHECK:     d2m.core_index(1)
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
        // CHECK:       d2m.reserve %cb0
        // CHECK:       d2m.dma_read %stream[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], <8>
        // CHECK:       d2m.dma_wait %{{.*}}
        // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       d2m.wait %cb0
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_load
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test 2: Case with remote_store - should be lowered to dma_write in datamovement region
  // CHECK-LABEL: func.func @test_remote_store
  func.func @test_remote_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: ^datamovement0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     d2m.core_index(0)
      // CHECK:     d2m.core_index(1)
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
        // CHECK:       d2m.reserve %cb1
        // CHECK:       d2m.dma_write %{{.*}}[%{{.*}}], %stream[%{{.*}}, %{{.*}}, %{{.*}}], <8>
        // CHECK:       d2m.dma_wait %{{.*}}
        // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       d2m.reserve %cb1
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_store
      %mem0 = d2m.reserve %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test 3: Case with both remote_load and remote_store
  // CHECK-LABEL: func.func @test_both_remote_load_store
  func.func @test_both_remote_load_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                         %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream0, %alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: ^datamovement0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     d2m.core_index(0)
      // CHECK:     d2m.core_index(1)
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
        // CHECK:       d2m.reserve %cb0
        // CHECK:       d2m.dma_read %stream[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], <8>
        // CHECK:       d2m.dma_wait %{{.*}}
        // CHECK:       d2m.reserve %cb2
        // CHECK:       d2m.dma_write %{{.*}}[%{{.*}}], %stream_2[%{{.*}}, %{{.*}}, %{{.*}}], <8>
        // CHECK:       d2m.dma_wait %{{.*}}
        // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       d2m.wait %cb0
      // CHECK:       d2m.reserve %cb2
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_load
      // CHECK-NOT: d2m.remote_store
      %in = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test 4: Case with matmul and multiple remote loads
  // CHECK-LABEL: func.func @test_matmul_multiple_loads
  func.func @test_matmul_multiple_loads(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
                                        %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: ^datamovement0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     d2m.core_index(0)
      // CHECK:     d2m.core_index(1)
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
        // CHECK:         d2m.reserve %cb0
        // CHECK:         d2m.dma_read %stream[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], <4>
        // CHECK:         d2m.dma_wait %{{.*}}
        // CHECK:         d2m.reserve %cb1
        // CHECK:         d2m.dma_read %stream_2[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], <4>
        // CHECK:         d2m.dma_wait %{{.*}}
        // CHECK:       } {d2m.outer_loop}
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:         d2m.wait %cb0
      // CHECK:         d2m.wait %cb1
      // CHECK:         d2m.reserve %cb2
      // CHECK:         d2m.tile_matmul_block
      // CHECK:       } {d2m.outer_loop}
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_load
      %lhs = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %rhs = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      "d2m.tile_matmul_block"(%lhs, %rhs, %out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    }
    return
  }

  // Test 5: Interleaved DRAM layout with grid 1x1 - should generate inner loops
  // This test uses an interleaved layout which creates a complex memory access pattern
  // that requires loops to read individual tile elements forming a single shard.
  // CHECK-LABEL: func.func @test_interleaved_dram_1x1_grid
  func.func @test_interleaved_dram_1x1_grid(%arg0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #dram>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<1x1>
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: ^datamovement0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     d2m.core_index(0)
      // CHECK:     d2m.core_index(1)
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       d2m.reserve %cb0
      // CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
      // CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args
      // CHECK:           scf.if
      // CHECK:             d2m.dma_read %stream[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], <{{.*}}>
      // CHECK:           }
      // CHECK:         }
      // CHECK:       }
      // CHECK:       d2m.dma_wait %{{.*}}
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       d2m.wait %cb0
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_load
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test 6: Multicast remote_load with 1x4 grid and reduction iterator
  // This tests the gather-multicast pattern where core 0 gathers data and multicasts to other cores
  // CHECK-LABEL: func.func @test_multicast_remote_load
  func.func @test_multicast_remote_load(%arg0: memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: ^datamovement0(%{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>, %{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>, %{{.*}}: !d2m.semaphore, %{{.*}}: !d2m.semaphore):
      // Verify semaphores are added for multicast synchronization
      // Verify core index check and multicast branching
      // CHECK-DAG: d2m.core_index(0)
      // CHECK-DAG: arith.cmpi eq
      // CHECK-DAG: scf.if
      // Verify sender path has gather DMA and multicast
      // CHECK-DAG: d2m.reserve %cb0
      // CHECK-DAG: d2m.dma_read %stream
      // CHECK-DAG: d2m.dma_wait
      // CHECK-DAG: d2m.semaphore_wait
      // CHECK-DAG: d2m.wait %cb0
      // CHECK-DAG: d2m.dma_write {{.*}} core[{{.*}}] mcast[{{.*}}]
      // CHECK-DAG: d2m.semaphore_set
      // Verify receiver path has semaphore operations
      // CHECK-DAG: d2m.semaphore_inc
      // CHECK-NOT: d2m.remote_load
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test 7: Multicast matmul with 2x6 grid
  // Tests multicast on LHS (mcast on dim 1). RHS has mcast[1,1] which is treated as unicast
  // since all multicast dimensions are 1 (no actual multicast needed).
  // CHECK-LABEL: func.func @test_multicast_matmul_2x6
  func.func @test_multicast_matmul_2x6(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<4x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x6x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x6x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x6x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<4x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #dram>, memref<2x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>) -> memref<4x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [1, 1, 4], grid = #ttcore.grid<2x6>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<4x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x6x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // Verify 2 semaphores are added (for LHS multicast only, RHS is unicast)
      // CHECK: ^datamovement0(%{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>, %{{.*}}: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>, %{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1{{.*}}>>, %{{.*}}: !d2m.semaphore, %{{.*}}: !d2m.semaphore):
      // Verify LHS multicast path with correct dimension check (core_index(1) == 0)
      // CHECK-DAG: arith.cmpi eq, %{{.*}}, %c0
      // CHECK-DAG: d2m.reserve %cb0
      // CHECK-DAG: d2m.dma_read %stream
      // CHECK-DAG: d2m.dma_write {{.*}} core[{{.*}}] mcast[{{.*}}]
      // Verify RHS unicast path (no multicast since mcast dimensions are all 1)
      // CHECK-DAG: d2m.reserve %cb1
      // CHECK-DAG: d2m.dma_read %stream_2
      // Verify semaphore synchronization operations exist for LHS
      // CHECK-DAG: d2m.semaphore_wait
      // CHECK-DAG: d2m.semaphore_set
      // CHECK-DAG: d2m.semaphore_inc
      // CHECK-NOT: d2m.remote_load
      %lhs = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %rhs = d2m.wait %cb1 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1_>
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}
