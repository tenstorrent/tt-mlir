// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dma-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

// Comprehensive test cases for d2m-insert-dma-ops pass
// These tests verify the same behavior as d2m-generic-generate-datamovement
// but expect operations to be inserted into the compute region wrapped in scf.execute_region

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Test 1: Simple add with non-stream operands (no DMA needed)
func.func @add(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>, %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>
  "d2m.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // CHECK: ^compute0(%cb0: !d2m.cb<{{.*}}>, %cb1: !d2m.cb<{{.*}}>, %cb2: !d2m.cb<{{.*}}>):
  // CHECK: scf.execute_region no_inline {
  // CHECK: d2m.reserve %cb0
  // CHECK: scf.yield
  // CHECK: }
  // CHECK: scf.execute_region no_inline {
  // CHECK: d2m.reserve %cb1
  // CHECK: scf.yield
  // CHECK: }
  ^bb0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    affine.for %arg5 = 0 to 2 {
      affine.for %arg6 = 0 to 4 {
        %0 = affine.load %mem0[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %1 = affine.load %mem1[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %2, %mem2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      }
    }
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>
}

#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#reduction = #ttcore.iterator_type<reduction>

// Test 2: Single core matmul with non-stream operands
func.func @matmul_single_core(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>, %arg1: memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  "d2m.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // CHECK: ^compute0(%cb0: !d2m.cb<{{.*}}>, %cb1: !d2m.cb<{{.*}}>, %cb2: !d2m.cb<{{.*}}>):
  // CHECK: scf.execute_region no_inline {
  // CHECK: d2m.reserve %cb0
  // CHECK: scf.yield
  // CHECK: }
  // CHECK: scf.execute_region no_inline {
  // CHECK: d2m.reserve %cb1
  // CHECK: scf.yield
  // CHECK: }
  ^bb0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%mem0, %mem1, %mem2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>,  #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
}

// Test 3: Single core matmul with stream operands (needs DMA)
func.func @matmul_single_core_stream(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
  %0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #l1_>
  %1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #l1_>
  // CHECK: d2m.generic
  // CHECK-NEXT: ins([[lhs:%[a-z0-9_]+]], [[rhs:%[a-z0-9_]+]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "d2m.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // CHECK: ^compute0(%cb0: !d2m.cb<{{.*}}>, %cb1: !d2m.cb<{{.*}}>, %cb2: !d2m.cb<{{.*}}>):
  // CHECK: scf.execute_region no_inline {
  // CHECK: d2m.reserve %cb0
  // CHECK: d2m.dma [[lhs]]
  // CHECK: d2m.dma_wait
  // CHECK: scf.yield
  // CHECK: }
  // CHECK: scf.execute_region no_inline {
  // CHECK: d2m.reserve %cb1
  // CHECK: d2m.dma [[rhs]]
  // CHECK: d2m.dma_wait
  // CHECK: scf.yield
  // CHECK: }
  ^bb0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%mem0, %mem1, %mem2) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.view<map(4)>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1_>
}

// Test 4: Multi-core matmul with streams and multicast (verifies semaphore inference)
func.func @matmul_multi_core_with_multicast(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>
  %cb0_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1_>
  %cb1_alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>
  %stream0 = "d2m.stream_layout"(%arg0, %cb0_alloc) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #ttcore.view<map(4)>, #l1_>
  %stream1 = "d2m.stream_layout"(%arg1, %cb1_alloc) : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>, memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>

  // Compute-only generic with stream operands - should have DMA ops inserted with multicast
  // Grid is 2x4, reduction is along d2 (K dimension), so both inputs need multicast
  "d2m.generic"(%stream0, %stream1, %alloc) <{block_factors = [1, 1, 4], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  // CHECK: d2m.generic
  // CHECK-NEXT: ins([[lhs:%[a-z0-9_]+]], [[rhs:%[a-z0-9_]+]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  // CHECK: ^compute0(%cb0: !d2m.cb<{{.*}}>, %cb1: !d2m.cb<{{.*}}>, %cb2: !d2m.cb<{{.*}}>, [[sem0:%[a-z0-9]+]]: !d2m.semaphore, [[sem1:%[a-z0-9]+]]: !d2m.semaphore):
  // CHECK: scf.execute_region no_inline {
  // CHECK: d2m.reserve %cb0
  // CHECK: scf.if
  // CHECK: d2m.dma [[lhs]]
  // CHECK: d2m.dma_wait
  // CHECK: d2m.semaphore_wait [[sem0]]
  // CHECK: d2m.dma {{%.*}}, {{%.*}} mcast
  // CHECK: d2m.dma_wait
  // CHECK: d2m.semaphore_set [[sem1]]
  // CHECK: } else {
  // CHECK: d2m.semaphore_inc [[sem0]]
  // CHECK: d2m.semaphore_wait [[sem1]]
  // CHECK: }
  // CHECK: scf.yield
  // CHECK: }
  // CHECK: scf.execute_region no_inline {
  // CHECK: d2m.reserve %cb1
  // CHECK: scf.if
  // CHECK: d2m.dma [[rhs]]
  // CHECK: d2m.dma_wait
  // CHECK: d2m.semaphore_wait [[sem0]]
  // CHECK: d2m.dma {{%.*}}, {{%.*}} mcast
  // CHECK: d2m.dma_wait
  // CHECK: d2m.semaphore_set [[sem1]]
  // CHECK: } else {
  // CHECK: d2m.semaphore_inc [[sem0]]
  // CHECK: d2m.semaphore_wait [[sem1]]
  // CHECK: }
  // CHECK: scf.yield
  // CHECK: }
  ^bb0(%cb0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem0 = d2m.wait %cb0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.wait %cb1 : !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%mem0, %mem1, %mem2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096, 1>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1_>
}
