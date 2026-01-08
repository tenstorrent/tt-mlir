// UNSUPPORTED: true
// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-load-store-ops --canonicalize -o %t %s
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
  // Test remote_load insertion for remote wait operations
  // CHECK-LABEL: func.func @test_remote_wait
  func.func @test_remote_wait(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: %[[IDX0:.*]] = d2m.iter_index(0)
      // CHECK: %[[IDX1:.*]] = d2m.iter_index(1)
      // CHECK: d2m.remote_load %cb0, %{{.*}}[%[[IDX0]], %[[IDX1]]]
      // CHECK: d2m.wait %cb0
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test remote_store insertion for remote reserve operations
  // CHECK-LABEL: func.func @test_remote_reserve
  func.func @test_remote_reserve(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK: %[[RESERVE:.*]] = d2m.reserve %cb1
      // CHECK: %[[IDX0:.*]] = d2m.iter_index(0)
      // CHECK: %[[IDX1:.*]] = d2m.iter_index(1)
      // CHECK: d2m.remote_store %{{.*}}[%[[IDX0]], %[[IDX1]]], %cb1
      %mem0 = d2m.reserve %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test local reserve and push insertion for local wait operations
  // CHECK-LABEL: func.func @test_local_wait
  func.func @test_local_wait(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // Local input CB: reserve and push should be inserted before wait
      // CHECK: %[[RESERVE:.*]] = d2m.reserve %cb0
      // CHECK: d2m.push %cb0
      // CHECK: d2m.wait %cb0
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test local wait and pop insertion for local reserve operations
  // CHECK-LABEL: func.func @test_local_reserve
  func.func @test_local_reserve(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // Local output CB: reserve is created, wait/pop/push should be inserted at end
      // CHECK: %[[RESERVE:.*]] = d2m.reserve %cb1
      %mem0 = d2m.reserve %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      // CHECK: d2m.wait %cb1
      // CHECK: d2m.pop %cb1
      // CHECK: d2m.push %cb1
    }
    return
  }

  // Test mixed remote and local operations
  // CHECK-LABEL: func.func @test_mixed_remote_local
  func.func @test_mixed_remote_local(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream0, %arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // Remote wait should have remote_load
      // CHECK: %[[IDX0:.*]] = d2m.iter_index(0)
      // CHECK: %[[IDX1:.*]] = d2m.iter_index(1)
      // CHECK: d2m.remote_load %cb0, %{{.*}}[%[[IDX0]], %[[IDX1]]]
      // CHECK: d2m.wait %cb0
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

      // Local input CB: reserve and push should be inserted before wait
      // CHECK: %[[RESERVE1:.*]] = d2m.reserve %cb1
      // CHECK: d2m.push %cb1
      // CHECK: d2m.wait %cb1
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test matmul with remote inputs
  // CHECK-LABEL: func.func @test_matmul_remote
  func.func @test_matmul_remote(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
                                 %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>

    d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      // Remote wait operations should have remote_load for both inputs
      // CHECK: d2m.iter_index(0)
      // CHECK: d2m.iter_index(2)
      // CHECK: d2m.remote_load %cb0, %{{.*}}[%{{.*}}, %{{.*}}]
      // CHECK: d2m.wait %cb0
      %lhs = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      // CHECK: d2m.iter_index(2)
      // CHECK: d2m.iter_index(1)
      // CHECK: d2m.remote_load %cb1, %{{.*}}[%{{.*}}, %{{.*}}]
      // CHECK: d2m.wait %cb1
      %rhs = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      // Local output CB: reserve is created, wait/pop/push should be inserted at end
      // CHECK: d2m.reserve %cb2
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      "d2m.tile_matmul_block"(%lhs, %rhs, %out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      // CHECK: d2m.wait %cb2
      // CHECK: d2m.pop %cb2
      // CHECK: d2m.push %cb2
    }
    return
  }

  // Test matmul with loop interchange
  // RUN: ttmlir-opt --ttcore-register-device --d2m-generic-apply-interchange="matmul-interchange=2,0,1" --d2m-insert-load-store-ops --loop-invariant-code-motion %s 2>&1 | FileCheck %s --check-prefix=CHECK-INTERCHANGE

  // CHECK-INTERCHANGE-LABEL: func.func @test_matmul_interchange
  func.func @test_matmul_interchange(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                      %arg1: memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>

    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      // CHECK-INTERCHANGE: d2m.wait %cb0
      %lhs = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      // CHECK-INTERCHANGE: d2m.wait %cb1
      %rhs = d2m.wait %cb1 : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
      // CHECK-INTERCHANGE: d2m.reserve %cb2
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      "d2m.tile_matmul_block"(%lhs, %rhs, %out) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      // CHECK-INTERCHANGE: d2m.wait %cb2
      // CHECK-INTERCHANGE: d2m.pop %cb2
      // CHECK-INTERCHANGE: d2m.push %cb2
    }
    return
  }

  // Test stream output operand generating remote_store
  // CHECK-LABEL: func.func @test_stream_output
  func.func @test_stream_output(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
                                 %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream_out = "d2m.stream_layout"(%arg1, %cb_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // Local input CB: reserve and push should be inserted before wait
      // CHECK: %[[RESERVE_IN:.*]] = d2m.reserve %cb0
      // CHECK: d2m.push %cb0
      // CHECK: d2m.wait %cb0
      %in = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

      // Output is remote stream, should have reserve followed by compute, then remote_store at end
      // CHECK: %[[RESERVE:.*]] = d2m.reserve %cb1
      %out = d2m.reserve %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

      // Simple copy operation
      // CHECK: affine.for
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 4 {
          %tile = affine.load %in[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
          affine.store %tile, %out[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        }
      }

      // Remote store should be at the very end, after all compute
      // CHECK: d2m.iter_index(0)
      // CHECK: d2m.iter_index(1)
      // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}], %cb1
    }
    return
  }

  // Test multicast remote_load with reduction iterator on multi-core grid
  // When an operand's indexing map has a reduction dimension mapped to grid,
  // the data should be gathered by core 0 and multicast to all other cores.
  // CHECK-LABEL: func.func @test_multicast_remote_load
  func.func @test_multicast_remote_load(%arg0: memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    // Input has grid 1x4, operand has grid 1x4
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // Grid 1x4 with iterator types [parallel, reduction]
    // Operand indexing map: (d0, d1) -> (d0, d1) - both dims map to grid
    // Dim 0 is parallel (grid dim 0 = 1), Dim 1 is reduction (grid dim 1 = 4)
    // Since grid dim 1 > 1 and iterator is reduction, multicast should be generated
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // Multicast remote_load: core[d2m.core_index(0), 1] mcast[1, 3]
      // Since dim 0 is parallel (grid 1), mcast start/shape = core_index(0), 1
      // Since dim 1 is reduction (grid 4), mcast start = 1, mcast shape = 3 (gridSize - 1)
      // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
      // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
      // CHECK-DAG: %[[CORE0:.*]] = d2m.core_index(0)
      // CHECK: d2m.remote_load %cb0, %{{.*}}[%{{.*}}, %{{.*}}] core[%[[CORE0]], %[[C1]]] mcast[%[[C1]], %[[C3]]]
      // CHECK: d2m.wait %cb0
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test multicast remote_load with 2x6 grid using matmul indexing maps
  // Generic grid: 2x6 (M x N), iterator types [parallel(M), parallel(N), reduction(K)]
  // LHS (in0) grid shape: 2x4 (M x K), indexing map (d0, d2)
  // RHS (in1) grid shape: 4x6 (K x N), indexing map (d2, d1)
  // Output grid shape: 2x6 (M x N), indexing map (d0, d1)
  // CHECK-LABEL: func.func @test_multicast_matmul_2x6
  func.func @test_multicast_matmul_2x6(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<4x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #dram>) {
    // Output: grid 2x6 (M x N)
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x6x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    // CB for LHS: shard shape matches blocked access
    %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x6x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    // CB for RHS: shard shape matches blocked access
    %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>
    // LHS stream: grid 2x4 (M x K)
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x6x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>
    // RHS stream: grid 4x6 (K x N)
    %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<4x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #dram>, memref<2x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>) -> memref<4x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #ttcore.view<map(4)>, #dram>

    // Generic grid 2x6, iterator types [parallel(M=d0), parallel(N=d1), reduction(K=d2)]
    // K dimension is blocked with block_factor 4
    // LHS map (d0, d2): grid dim 0 = d0 (parallel), grid dim 1 = d2 (reduction) -> mcast on dim 1
    // RHS map (d2, d1): grid dim 0 = d2 (reduction), grid dim 1 = d1 (parallel) -> mcast on dim 0
    d2m.generic {block_factors = [1, 1, 4], grid = #ttcore.grid<2x6>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<4x6x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x6x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      // LHS remote_load: map (d0, d2) with iterators [parallel, parallel, reduction]
      // Grid dim 0 = d0 (parallel) -> mcast start = core_index(0), mcast shape = 1
      // Grid dim 1 = d2 (reduction) -> mcast start = 1, mcast shape = 5 (gridSize - 1 = 6 - 1)
      // CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
      // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
      // CHECK-DAG: %[[CORE0:.*]] = d2m.core_index(0)
      // CHECK: d2m.remote_load %cb0, %{{.*}}[%{{.*}}, %{{.*}}] core[%[[CORE0]], %[[C1]]] mcast[%[[C1]], %[[C5]]]
      // CHECK: d2m.wait %cb0
      %lhs = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

      // RHS remote_load: map (d2, d1) with iterators [parallel, parallel, reduction]
      // Grid dim 0 = d2 (reduction) -> mcast start = 1, mcast shape = 1 (gridSize - 1 = 2 - 1)
      // Grid dim 1 = d1 (parallel) -> mcast start = core_index(1), mcast shape = 1
      // CHECK: %[[CORE1:.*]] = d2m.core_index(1)
      // CHECK: d2m.remote_load %cb1, %{{.*}}[%{{.*}}, %{{.*}}] core[%[[C1]], %[[CORE1]]] mcast[%[[C1]], %[[C1]]]
      // CHECK: d2m.wait %cb1
      %rhs = d2m.wait %cb1 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1_>

      // Local output - no multicast (all parallel)
      // CHECK: d2m.reserve %cb2
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

}
