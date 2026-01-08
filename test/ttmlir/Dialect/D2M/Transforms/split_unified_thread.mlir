// UNSUPPORTED: true
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-load-store-ops --d2m-generate-outer-loops --d2m-split-unified-thread -o %t %s
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
  // Test 1: Simple case with remote_load - should split into datamovement and compute regions
  // Verifies loops are preserved in both regions and remote_load is in datamovement region
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
      // CHECK:       arith.addi
      // CHECK:       arith.addi
      // CHECK:       d2m.remote_load %cb0, %stream[%{{.*}}, %{{.*}}]
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.wait
      // CHECK-NOT: d2m.reserve
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       d2m.wait %cb0
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_load
      // CHECK-NOT: d2m.remote_store
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test 2: Case with remote_store - should split correctly with loops preserved
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
      // CHECK:       arith.addi
      // CHECK:       arith.addi
      // CHECK:       d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}], %cb1
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.wait
      // CHECK-NOT: d2m.reserve
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       %{{.*}} = d2m.reserve %cb1
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_load
      // CHECK-NOT: d2m.remote_store
      %mem0 = d2m.reserve %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test 3: Case with both remote_load and remote_store - should split correctly with loops preserved
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
      // CHECK:       arith.addi
      // CHECK:       arith.addi
      // CHECK:       d2m.remote_load %cb0, %stream[%{{.*}}, %{{.*}}]
      // CHECK:       d2m.remote_store %stream_2[%{{.*}}, %{{.*}}], %cb2
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.wait
      // CHECK-NOT: d2m.reserve
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       d2m.wait %cb0
      // CHECK:       %{{.*}} = d2m.reserve %cb2
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_load
      // CHECK-NOT: d2m.remote_store
      %in = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test 4: Case with matmul and multiple remote loads - should split correctly with 3 nested loops
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
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:         d2m.remote_load %cb0, %stream[%{{.*}}, %{{.*}}]
      // CHECK:         d2m.remote_load %cb1, %stream_2[%{{.*}}, %{{.*}}]
      // CHECK:       } {d2m.outer_loop}
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.wait
      // CHECK-NOT: d2m.reserve
      // CHECK-NOT: d2m.tile_matmul_block
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
      // CHECK-NOT: d2m.remote_store
      %lhs = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %rhs = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      "d2m.tile_matmul_block"(%lhs, %rhs, %out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    }
    return
  }

  // Test 5: Case with local operands only (no remote_load/store inserted) - should still split
  // but datamovement region will be empty (only loops and terminator)
  // CHECK-LABEL: func.func @test_local_only
  func.func @test_local_only(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: ^datamovement0
      // CHECK-NOT: d2m.remote_load
      // CHECK-NOT: d2m.remote_store
      // CHECK: ^compute0
      // CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
      // CHECK:       d2m.wait %cb0
      // CHECK:     } {d2m.outer_loop}
      // CHECK:   } {d2m.outer_loop}
      // CHECK-NOT: d2m.remote_load
      // CHECK-NOT: d2m.remote_store
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}
