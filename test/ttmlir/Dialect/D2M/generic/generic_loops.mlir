// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-generate-loops -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttcore.memory_space<dram>
#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

func.func @unary2x4(%arg0: memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
  "d2m.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^datamovement0(%cb0: !d2m.cb<memref<128x192xf32, #l1_>>, %cb1: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<128x192xf32, #l1>> -> memref<128x192xf32, #l1>
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<128x192xf32, #l1_>> -> memref<128x192xf32, #l1_>
  }, {
  ^datamovement1(%cb0: !d2m.cb<memref<128x192xf32, #l1_>>, %cb1: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<4x6x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^compute(%cb0: !d2m.cb<memref<128x192xf32, #l1_>>, %cb1: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<128x192xf32, #l1>> -> memref<128x192xf32, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<4x6x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: "d2m.tile_tilize_block"
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<128x192xf32, #l1_>> -> memref<128x192xf32, #l1_>
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_tilize_block"(%mem0, %mem1) : (memref<128x192xf32, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
}

func.func @binary1x1(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  "d2m.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^datamovement1(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^datamovement2(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %mem2 = d2m.wait %cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^compute(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        %0 = affine.load %mem0[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %1 = affine.load %mem1[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %2, %mem2[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      }
    }
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}

func.func @matmul1x1(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, %arg1: memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  "d2m.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^datamovement1(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<4x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1>
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^datamovement2(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    %mem2 = d2m.wait %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^compute(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<4x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: "d2m.tile_matmul_block"
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%mem0, %mem1, %mem2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @matmul_outer_k(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %stream = "d2m.stream_layout"(%arg0, %alloc_0) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  %stream_2 = "d2m.stream_layout"(%arg1, %alloc_1) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  "d2m.generic"(%stream, %stream_2, %alloc) <{block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %[[I:.*]] = %c0 to %c2
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: d2m.dma %{{.*}}[%c0, %[[I]]]
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    %c0 = arith.constant 0 : index
    %k = d2m.iter_index(2) : index
    %tx = d2m.dma %stream[%c0, %k], %mem0 : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^datamovement1(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %[[I:.*]] = %c0 to %c2
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: d2m.dma %{{.*}}[%[[I]], %c0]
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    %c0 = arith.constant 0 : index
    %k = d2m.iter_index(2) : index
    %tx = d2m.dma %stream_2[%k, %c0], %mem1 : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^datamovement2(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c2
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    %mem2 = d2m.wait %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^compute(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c2
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: "d2m.tile_matmul_block"
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%mem0, %mem1, %mem2) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %stream = "d2m.stream_layout"(%arg0, %alloc_0) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  %stream_2 = "d2m.stream_layout"(%arg1, %alloc_1) : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>, memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #ttcore.view<map(4)>, #l1_>
  "d2m.generic"(%stream, %stream_2, %alloc) <{block_factors = [1, 1, 4], grid = #ttcore.grid<2x4>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c4
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<4x6x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1>
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^datamovement1(%cb0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c4
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<6x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1>
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^datamovement2(%cb0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c4
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<4x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1>
    %mem2 = d2m.wait %cb2 : !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1_>
  }, {
  ^compute(%cb0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c4
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<4x6x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.wait %{{.*}} : <memref<6x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: %{{.*}} = d2m.reserve %{{.*}} : <memref<4x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1>
    // CHECK-NEXT: "d2m.tile_matmul_block"
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%mem0, %mem1, %mem2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #ttcore.view<map(4)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}
