// RUN: ttmlir-opt --ttcore-register-device --d2m-generate-outer-loops -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_add = affine_map<(d0)[s0] -> (d0 + s0)>

module {
  // Test 1: Simple case with block_factors [1, 1] - should generate 2 nested loops
  // Verifies loop generation and loop bounds
  // CHECK-LABEL: func.func @test_simple_1x1
  func.func @test_simple_1x1(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: block_factors = [1, 1]
  // CHECK-NOT: d2m.block_index
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
  // CHECK-DAG: %{{.*}} = d2m.get_block_factor(0) : index
  // CHECK-DAG: %{{.*}} = d2m.get_block_factor(1) : index
  // CHECK: affine.for %[[I:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:   affine.for %[[J:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:     %[[OFF0:.*]] = d2m.block_offset(0)
  // CHECK-NEXT:     %[[IDX0:.*]] = affine.apply #{{.*}}(%[[I]])[%[[OFF0]]]
  // CHECK-NEXT:     %[[OFF1:.*]] = d2m.block_offset(1)
  // CHECK-NEXT:     %[[IDX1:.*]] = affine.apply #{{.*}}(%[[J]])[%[[OFF1]]]
  // CHECK-NEXT:     %{{.*}} = memref.alloc
  // CHECK-NEXT:     %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%[[IDX0]], %[[IDX1]]] : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
  // CHECK-NEXT:     %{{.*}} = memref.alloc
  // CHECK-NEXT:     %{{.*}} = d2m.remote_store %{{.*}}[%[[IDX0]], %[[IDX1]]] %{{.*}} : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
    // CHECK-NEXT:   } {d2m.blocking_loop = 1
    // CHECK-NEXT: } {d2m.blocking_loop = 0
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %idx0 = d2m.block_index(0) : index
      %idx1 = d2m.block_index(1) : index
      %buffer_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %in = d2m.remote_load %buffer_in %stream[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %result = d2m.remote_store %alloc[%idx0, %idx1] %buffer : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    }
    return
  }

  // Test 2: Case with block_factors [1, 1, 2] - should generate 3 nested loops
  // Verifies loop generation with reduction dimension and remote_load insertion
  // CHECK-LABEL: func.func @test_matmul_1x1x2
  func.func @test_matmul_1x1x2(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
                                 %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) <{remapping = #map4}> : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) <{remapping = #map4}> : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<4>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: block_factors = [1, 1, 2]
  // CHECK-NOT: d2m.block_index
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %{{.*}}: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %{{.*}}: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>):
  // CHECK-DAG: %{{.*}} = d2m.get_block_factor(0) : index
  // CHECK-DAG: %{{.*}} = d2m.get_block_factor(1) : index
  // CHECK-DAG: %{{.*}} = d2m.get_block_factor(2) : index
  // CHECK: affine.for %[[I:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:   affine.for %[[J:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:     affine.for %{{.*}} = 0 to %{{.*}} {
  // CHECK-NEXT:       %[[OFF0:.*]] = d2m.block_offset(0)
  // CHECK-NEXT:       %[[IDX0:.*]] = affine.apply #{{.*}}(%[[I]])[%[[OFF0]]]
  // CHECK-NEXT:       %[[OFF1:.*]] = d2m.block_offset(1)
  // CHECK-NEXT:       %[[IDX1:.*]] = affine.apply #{{.*}}(%[[J]])[%[[OFF1]]]
  // CHECK-NEXT:       %{{.*}} = memref.alloc
  // CHECK-NEXT:       %{{.*}} = memref.alloc
  // CHECK-NEXT:       %{{.*}} = memref.alloc
  // CHECK-NEXT:       %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%[[IDX0]], %[[IDX1]]] : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
  // CHECK-NEXT:       %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%[[IDX0]], %[[IDX1]]] : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
    // CHECK-NEXT:       "d2m.tile_matmul_block"
    // CHECK-NEXT:       %{{.*}} = d2m.remote_store %{{.*}}[%[[IDX0]], %[[IDX1]]] %{{.*}} : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
    // CHECK-NEXT:     } {d2m.blocking_loop = 2
    // CHECK-NEXT:   } {d2m.blocking_loop = 1
    // CHECK-NEXT: } {d2m.blocking_loop = 0
    d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<4>, #dram>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %idx0 = d2m.block_index(0) : index
      %idx1 = d2m.block_index(1) : index
      %buffer_lhs = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %buffer_rhs = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %buffer_out = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %lhs = d2m.remote_load %buffer_lhs %stream0[%idx0, %idx1] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %rhs = d2m.remote_load %buffer_rhs %stream1[%idx0, %idx1] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      "d2m.tile_matmul_block"(%lhs, %rhs, %buffer_out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      %result = d2m.remote_store %alloc[%idx0, %idx1] %buffer_out : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    }
    return
  }

  // Test 3: Test that outer_loop attribute is set on all loops
  // CHECK-LABEL: func.func @test_outer_loop_attribute
  func.func @test_outer_loop_attribute(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: block_factors = [1, 1]
  // CHECK-NOT: d2m.block_index
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
  // CHECK-DAG: %{{.*}} = d2m.get_block_factor(0) : index
  // CHECK-DAG: %{{.*}} = d2m.get_block_factor(1) : index
  // CHECK: affine.for %[[I:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:   affine.for %[[J:.*]] = 0 to %{{.*}} {
  // CHECK-NEXT:     %[[OFF0:.*]] = d2m.block_offset(0)
  // CHECK-NEXT:     %[[IDX0:.*]] = affine.apply #{{.*}}(%[[I]])[%[[OFF0]]]
  // CHECK-NEXT:     %[[OFF1:.*]] = d2m.block_offset(1)
  // CHECK-NEXT:     %[[IDX1:.*]] = affine.apply #{{.*}}(%[[J]])[%[[OFF1]]]
  // CHECK-NEXT:     %{{.*}} = memref.alloc
  // CHECK-NEXT:     %{{.*}} = d2m.remote_load %{{.*}} %{{.*}}[%[[IDX0]], %[[IDX1]]] : memref<{{.*}}>, memref<{{.*}}> -> memref<{{.*}}>
    // CHECK-NEXT:   } {d2m.blocking_loop = 1
    // CHECK-NEXT: } {d2m.blocking_loop = 0
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %idx0 = d2m.block_index(0) : index
      %idx1 = d2m.block_index(1) : index
      %buffer_mem = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %mem0 = d2m.remote_load %buffer_mem %stream[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

}
