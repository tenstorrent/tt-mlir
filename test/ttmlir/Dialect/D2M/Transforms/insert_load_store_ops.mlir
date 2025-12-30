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

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
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

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>) {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
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

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
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

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
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

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%stream0, %arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
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

    d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%stream0, %stream1 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
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

    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%stream0, %stream1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
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

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>) {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
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
}
