// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-multicast-loads %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  // Test lowering of high-level multicast form (mcast[dims]) to low-level form (core[...] mcast[...])
  // Grid: 2x4 with iterator types [parallel, reduction]
  // Multicast dimension: 1 (reduction dimension with grid size 4)
  // Expected: core[core_index(0), 0] mcast[1, 4]
  // CHECK-LABEL: func.func @test_multicast_single_dim
  func.func @test_multicast_single_dim(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // High-level mcast form: mcast on dim 1 (reduction dimension)
      // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
      // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
      // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
      // CHECK-DAG: %[[CORE0:.*]] = d2m.core_index(0)
      // CHECK: %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}] mcore[%[[CORE0]], %[[C0]]] mshape[%[[C1]], %[[C4]]]
      %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %load_result = d2m.remote_load %buffer %stream[%c0, %c1] mcast[%c0] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test lowering of high-level multicast form with multicast on first dimension
  // Grid: 4x2 with iterator types [reduction, parallel]
  // Multicast dimension: 0 (reduction dimension with grid size 4)
  // Expected: core[0, core_index(1)] mcast[4, 1]
  // CHECK-LABEL: func.func @test_multicast_first_dim
  func.func @test_multicast_first_dim(%arg0: memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) <{remapping = #map4}> : (memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x2>, indexing_maps = [#map, #map], iterator_types = [#reduction, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // High-level mcast form: mcast on dim 0 (reduction dimension)
      // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
      // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
      // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
      // CHECK-DAG: %[[CORE1:.*]] = d2m.core_index(1)
      // CHECK: %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}] mcore[%[[C0]], %[[CORE1]]] mshape[%[[C4]], %[[C1]]]
      %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %load_result = d2m.remote_load %buffer %stream[%c0, %c1] mcast[%c1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

  // Test that multicast on grid size 1 dimensions gets stripped to unicast
  // Grid: 1x4 with iterator types [reduction, parallel]
  // Multicast dimension: 0 (but grid size is 1, so should become unicast)
  // Expected: unicast remote_load (no core/mcast)
  // CHECK-LABEL: func.func @test_multicast_unicast_strip
  func.func @test_multicast_unicast_strip(%arg0: memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) <{remapping = #map4}> : (memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x4>, indexing_maps = [#map, #map], iterator_types = [#reduction, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // High-level mcast form on dim 0 with grid size 1 - should become unicast
      // CHECK: %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%{{.*}}, %{{.*}}] :
      // CHECK-NOT: mcore[
      // CHECK-NOT: mshape[
      %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %load_result = d2m.remote_load %buffer %stream[%c0, %c1] mcast[%c1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<1x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }
}
