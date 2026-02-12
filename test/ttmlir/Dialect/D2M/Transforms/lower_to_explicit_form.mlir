// RUN: ttmlir-opt --ttcore-register-device --d2m-generate-outer-loops --d2m-lower-to-explicit-form -o %t %s
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
  // Test 1: Simple case with block_factors [1, 1]
  // Verifies that affine.for loops are replaced with scf.for using constant bounds
  // and all symbolic index ops are made explicit.
  // CHECK-LABEL: func.func @test_simple_1x1
  func.func @test_simple_1x1(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %cb_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>

    // CHECK: d2m.generic
    // CHECK-SAME: block_factors = [1, 1]
    // CHECK-DAG: d2m.core_index(0)
    // CHECK-DAG: d2m.core_index(1)
    // CHECK-DAG: arith.addi
    // CHECK-NOT: d2m.block_offset
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %idx0 = d2m.block_index(0) : index
      %idx1 = d2m.block_index(1) : index
      %buffer_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %in = d2m.remote_load %buffer_in %stream[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<map(4)>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %result = d2m.remote_store %alloc[%idx0, %idx1] %buffer : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #l1_> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>
    }
    return
  }

  // Test 2: Case with block_factors [1, 1, 2] - 3 nested loops with different bounds
  // Verifies that the third dimension gets upper bound of 2
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
    // CHECK-DAG: d2m.core_index(0)
    // CHECK-DAG: d2m.core_index(1)
    // CHECK-NOT: d2m.core_index(2)
    // CHECK-DAG: arith.addi
    // CHECK-NOT: d2m.block_offset
    d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %idx0 = d2m.block_index(0) : index
      %idx1 = d2m.block_index(1) : index
      %idx2 = d2m.block_index(2) : index
      %buffer_lhs = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %buffer_rhs = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %buffer_out = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %lhs = d2m.remote_load %buffer_lhs %stream0[%idx0, %idx2] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %rhs = d2m.remote_load %buffer_rhs %stream1[%idx2, %idx1] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #dram> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      "d2m.tile_matmul_block"(%lhs, %rhs, %buffer_out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
      %result = d2m.remote_store %alloc[%idx0, %idx1] %buffer_out : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
    }
    return
  }
}
