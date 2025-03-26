// RUN: ttmlir-opt --tt-register-device --ttir-generic-generate-loops %s | FileCheck %s

#dram = #tt.memory_space<dram>
#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>

func.func @unary2x4(%arg0: memref<2x4x128x192xf32, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^datamovement0(%cb0: memref<128x192xf32, #l1_>, %cb1: memref<4x6x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.yield %cb0 : (memref<128x192xf32, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<128x192xf32, #l1_>, %cb1: memref<4x6x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.await %cb1 : (memref<4x6x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<128x192xf32, #l1_>, %cb1: memref<4x6x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.await %cb0 : (memref<128x192xf32, #l1_>)
    "ttir.tile_tilize_block"(%cb0, %cb1) : (memref<128x192xf32, #l1_>, memref<4x6x!tt.tile<32x32, f32>, #l1_>) -> ()
    ttir.yield %cb1 : (memref<4x6x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<2x4x128x192xf32, #l1_>, memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
}

func.func @binary1x1(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.yield %cb0 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.yield %cb1 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement2(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.await %cb2 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.await %cb0, %cb1 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<2x4x!tt.tile<32x32, f32>, #l1_>)
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        %0 = affine.load %cb0[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %1 = affine.load %cb1[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        affine.store %2, %cb2[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
      }
    }
    ttir.yield %cb2 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}

func.func @matmul1x1(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.yield %cb0 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.yield %cb1 : (memref<4x2x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement2(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.await %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    ttir.await %cb0, %cb1 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>)
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
    ttir.yield %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
}

func.func @matmul_outer_k(%arg0: memref<1x2x2x2x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<2x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  %stream = "ttir.stream_layout"(%arg0, %alloc_0) : (memref<1x2x2x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 8192 + d3 * 4096)>, #l1_>
  %stream_2 = "ttir.stream_layout"(%arg1, %alloc_1) : (memref<2x1x2x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> memref<2x1x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 8192 + d3 * 4096)>, #l1_>
  "ttir.generic"(%stream, %stream_2, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %[[I:.*]] = %c0 to %c2
    // CHECK-NEXT: ttir.dma %{{.*}}[%c0, %[[I]]]
    %c0 = arith.constant 0 : index
    %k = ttir.iter_index(2) : index
    %tx = ttir.dma %stream[%c0, %k], %cb0 : (memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 8192 + d3 * 4096)>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
    ttir.dma_wait %tx
    ttir.yield %cb0 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %[[I:.*]] = %c0 to %c2
    // CHECK-NEXT: ttir.dma %{{.*}}[%[[I]], %c0]
    %c0 = arith.constant 0 : index
    %k = ttir.iter_index(2) : index
    %tx = ttir.dma %stream_2[%k, %c0], %cb1 : (memref<2x1x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 8192 + d3 * 4096)>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
    ttir.dma_wait %tx
    ttir.yield %cb1 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement2(%cb0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c2
    ttir.await %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c2
    ttir.await %cb0, %cb1 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>)
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
    ttir.yield %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<1x2x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 8192 + d3 * 4096)>, #l1_>, memref<2x1x2x2x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 8192 + d3 * 4096)>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
}

func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!tt.tile<32x32, f32>, #l1_>
  %stream = "ttir.stream_layout"(%arg0, %alloc_0) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>, memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 24576 + d3 * 4096)>, #l1_>
  %stream_2 = "ttir.stream_layout"(%arg1, %alloc_1) : (memref<4x4x6x8x!tt.tile<32x32, f32>, #l1_>, memref<2x4x6x8x!tt.tile<32x32, f32>, #l1_>) -> memref<4x4x6x8x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 32768 + d3 * 4096)>, #l1_>
  "ttir.generic"(%stream, %stream_2, %alloc) <{grid = #tt.grid<2x4>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!tt.tile<32x32, f32>, #l1_>, %sem0: !ttir.semaphore, %sem1: !ttir.semaphore, %sem2: !ttir.semaphore, %sem3: !ttir.semaphore):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c4
    ttir.yield %cb0 : (memref<4x6x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!tt.tile<32x32, f32>, #l1_>, %sem0: !ttir.semaphore, %sem1: !ttir.semaphore, %sem2: !ttir.semaphore, %sem3: !ttir.semaphore):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c4
    ttir.yield %cb1 : (memref<6x8x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement2(%cb0: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!tt.tile<32x32, f32>, #l1_>, %sem0: !ttir.semaphore, %sem1: !ttir.semaphore, %sem2: !ttir.semaphore, %sem3: !ttir.semaphore):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c4
    ttir.await %cb2 : (memref<4x8x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<4x6x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!tt.tile<32x32, f32>, #l1_>, %sem0: !ttir.semaphore, %sem1: !ttir.semaphore, %sem2: !ttir.semaphore, %sem3: !ttir.semaphore):
    // CHECK: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c1
    // CHECK-NEXT: scf.for %{{.*}} = %c0 to %c4
    ttir.await %cb0, %cb1 : (memref<4x6x!tt.tile<32x32, f32>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #l1_>)
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x6x!tt.tile<32x32, f32>, #l1_>, memref<6x8x!tt.tile<32x32, f32>, #l1_>, memref<4x8x!tt.tile<32x32, f32>, #l1_>) -> ()
    ttir.yield %cb2 : (memref<4x8x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<2x4x4x6x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 24576 + d3 * 4096)>, #l1_>, memref<4x4x6x8x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 32768 + d3 * 4096)>, #l1_>, memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!tt.tile<32x32, f32>, #l1_>
}
