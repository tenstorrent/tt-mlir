// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-load-store-ops --d2m-generate-outer-loops -o %t %s
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
  // Test 1: Simple case with block_factors [1, 1] - should generate 2 nested loops
  // Verifies loop generation and loop bounds
  // CHECK-LABEL: func.func @test_simple_1x1
  func.func @test_simple_1x1(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: block_factors = [1, 1]
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #{{.*}}>>, %{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #{{.*}}>>):
    // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-NEXT: scf.for %{{.*}} = %[[C0]] to %[[C1]] step %[[C1]] {
    // CHECK-NEXT:   scf.for %{{.*}} = %[[C0]] to %[[C1]] step %[[C1]] {
    // CHECK-NEXT:     d2m.wait
    // CHECK-NEXT:     d2m.reserve
    // CHECK-NEXT:   } {d2m.outer_loop}
    // CHECK-NEXT: } {d2m.outer_loop}
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %in = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %out = d2m.reserve %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
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
    %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: block_factors = [1, 1, 2]
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #{{.*}}>>, %{{.*}}: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #{{.*}}>>, %{{.*}}: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #{{.*}}>>):
    // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-NEXT: scf.for %[[I:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
    // CHECK-NEXT:   %[[CORE0:.*]] = d2m.core_index(0) : index
    // CHECK-NEXT:   %[[CORE1:.*]] = d2m.core_index(1) : index
    // CHECK-NEXT:   scf.for %[[J:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
    // CHECK-NEXT:     scf.for %[[K:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
    // CHECK-NEXT:       %{{.*}} = arith.addi %[[CORE0]], %[[I]] : index
    // CHECK-NEXT:       %{{.*}} = d2m.remote_load %{{.*}}, %{{.*}}[%{{.*}}, %[[K]]]
    // CHECK-NEXT:       %{{.*}} = d2m.wait
    // CHECK-NEXT:       %{{.*}} = arith.addi %[[CORE1]], %[[J]] : index
    // CHECK-NEXT:       %{{.*}} = d2m.remote_load %{{.*}}, %{{.*}}[%[[K]], %{{.*}}]
    // CHECK-NEXT:       %{{.*}} = d2m.wait
    // CHECK-NEXT:       %{{.*}} = d2m.reserve
    // CHECK-NEXT:       d2m.tile_matmul_block
    // CHECK-NEXT:       %{{.*}} = d2m.wait
    // CHECK-NEXT:       d2m.pop
    // CHECK-NEXT:       d2m.push
    // CHECK-NEXT:     } {d2m.outer_loop}
    // CHECK-NEXT:   } {d2m.outer_loop}
    // CHECK-NEXT: } {d2m.outer_loop}
    d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %lhs = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %rhs = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %out = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      "d2m.tile_matmul_block"(%lhs, %rhs, %out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    }
    return
  }

  // Test 3: Test that outer_loop attribute is set on all loops
  // CHECK-LABEL: func.func @test_outer_loop_attribute
  func.func @test_outer_loop_attribute(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: block_factors = [1, 1]
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #{{.*}}>>, %{{.*}}: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #{{.*}}>>):
    // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-NEXT: scf.for %{{.*}} = %[[C0]] to %[[C1]] step %[[C1]] {
    // CHECK-NEXT:   scf.for %{{.*}} = %[[C0]] to %[[C1]] step %[[C1]] {
    // CHECK-NEXT:     d2m.wait
    // CHECK-NEXT:   } {d2m.outer_loop}
    // CHECK-NEXT: } {d2m.outer_loop}
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }
    return
  }

}
