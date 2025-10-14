// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-lower-dmas --cse -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttcore.memory_space<dram>
#l1_ = #ttcore.memory_space<l1>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

func.func @matmul_single_core_stream(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %stream = "d2m.stream_layout"(%arg0, %alloc_0) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  %stream_2 = "d2m.stream_layout"(%arg1, %alloc_1) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  "d2m.generic"(%stream, %stream_2, %alloc) <{block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    // CHECK-DAG: [[iter0:%.*]] = d2m.iter_index(0)
    // CHECK-DAG: [[iter2:%.*]] = d2m.iter_index(2)
    // CHECK-DAG: [[core0:%.*]] = d2m.core_index(0)
    // CHECK-DAG: [[offset:%.*]] = arith.addi [[core0]], [[iter0]]
    // CHECK: d2m.dma_read %stream{{[_0-9]*}}[[[offset]], [[iter2]], %c0]
    %tx = d2m.dma %stream<#map1>, %cb0 : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
    d2m.dma_wait %tx
    d2m.yield %cb0 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    // CHECK-DAG: [[iter2:%.*]] = d2m.iter_index(2)
    // CHECK-DAG: [[iter1:%.*]] = d2m.iter_index(1)
    // CHECK-DAG: [[core1:%.*]] = d2m.core_index(1)
    // CHECK-DAG: [[offset:%.*]] = arith.addi [[core1]], [[iter1]]
    // CHECK: d2m.dma_read %stream{{[_0-9]*}}[[[iter2]], [[offset]], %c0]
    %tx = d2m.dma %stream_2<#map2>, %cb1 : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
    d2m.dma_wait %tx
    d2m.yield %cb1 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement2(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    d2m.await %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    d2m.await %cb0, %cb1 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    d2m.yield %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @matmul_single_core_transpose(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  %stream = "d2m.stream_layout"(%arg0, %alloc_0) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  %stream_2 = "d2m.stream_layout"(%arg1, %alloc_1) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>, #l1_>
  "d2m.generic"(%stream, %stream_2, %alloc) <{block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    // CHECK-DAG: [[iter0:%.*]] = d2m.iter_index(0)
    // CHECK-DAG: [[iter2:%.*]] = d2m.iter_index(2)
    // CHECK-DAG: [[core0:%.*]] = d2m.core_index(0)
    // CHECK-DAG: [[offset:%.*]] = arith.addi [[core0]], [[iter0]]
    // CHECK: d2m.dma_read %stream{{[_0-9]*}}[[[offset]], [[iter2]], %c0]
    %tx = d2m.dma %stream<#map1>, %cb0 : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
    d2m.dma_wait %tx
    d2m.yield %cb0 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    // CHECK-DAG: [[iter2:%.*]] = d2m.iter_index(2)
    // CHECK-DAG: [[core1:%.*]] = d2m.core_index(1)
    // CHECK: d2m.null_tx
    // CHECK-NEXT: scf.for [[for_iter_i:%[a-zA-Z0-9]*]] =
    // CHECK-NEXT: scf.for [[for_iter_j:%[a-zA-Z0-9]*]] =
    // CHECK: d2m.dma_read %stream{{[_0-9]*}}
    // CHECK: scf.yield
    // CHECK: scf.yield
    %tx = d2m.dma %stream_2<#map2>, %cb1 : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
    d2m.dma_wait %tx
    d2m.yield %cb1 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement2(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    d2m.await %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    d2m.await %cb0, %cb1 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
    "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    d2m.yield %cb2 : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>)
  }) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %stream = "d2m.stream_layout"(%arg0, %alloc_0) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  %stream_2 = "d2m.stream_layout"(%arg1, %alloc_1) : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>, memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  "d2m.generic"(%stream, %stream_2, %alloc) <{block_factors = [1, 1, 4], grid = #ttcore.grid<2x4>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %core0 = d2m.core_index(0) : index
    %core1 = d2m.core_index(1) : index
    %0 = arith.cmpi eq, %core1, %c0 : index
    scf.if %0 {
      // CHECK-DAG: [[iter0:%.*]] = d2m.iter_index(0)
      // CHECK-DAG: [[iter2:%.*]] = d2m.iter_index(2)
      // CHECK-DAG: [[core0:%.*]] = d2m.core_index(0)
      // CHECK-DAG: [[offset:%.*]] = arith.addi [[core0]], [[iter0]]
      // CHECK: d2m.dma_read %stream{{[_0-9]*}}[[[offset]], [[iter2]], %c0]
      %tx = d2m.dma %stream<#map1>, %cb0 : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      d2m.dma_wait %tx
      d2m.semaphore_wait %sem0, %c3 reset %c0
      %tx_3 = d2m.dma %cb0, %cb0 core[%core0, %c0] mcast[%c1, %c4] : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      d2m.dma_wait %tx_3
      d2m.semaphore_set %sem1, %c1, core[%core0, %c0] mcast[%c1, %c4]
    } else {
      d2m.semaphore_inc %sem0, %c1, core[%core0, %c0]
      d2m.semaphore_wait %sem1, %c1 reset %c0
    }
    d2m.yield %cb0 : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %core0 = d2m.core_index(0) : index
    %0 = arith.cmpi eq, %core0, %c0 : index
    %core1 = d2m.core_index(1) : index
    scf.if %0 {
      // CHECK-DAG: [[iter2:%.*]] = d2m.iter_index(2)
      // CHECK-DAG: [[iter1:%.*]] = d2m.iter_index(1)
      // CHECK-DAG: [[core1:%.*]] = d2m.core_index(1)
      // CHECK-DAG: [[offset:%.*]] = arith.addi [[core1]], [[iter1]]
      // CHECK: d2m.dma_read %stream{{[_0-9]*}}[[[iter2]], [[offset]], %c0]
      %tx = d2m.dma %stream_2<#map2>, %cb1 : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      d2m.dma_wait %tx
      d2m.semaphore_wait %sem2, %c1 reset %c0
      %tx_3 = d2m.dma %cb1, %cb1 core[%c0, %core1] mcast[%c2, %c1] : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      d2m.dma_wait %tx_3
      d2m.semaphore_set %sem3, %c1, core[%c0, %core1] mcast[%c2, %c1]
    } else {
      d2m.semaphore_inc %sem2, %c1, core[%c0, %core1]
      d2m.semaphore_wait %sem3, %c1 reset %c0
    }
    d2m.yield %cb1 : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement2(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    d2m.await %cb2 : (memref<4x8x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    d2m.await %cb0, %cb1 : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>)
    "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    d2m.yield %cb2 : (memref<4x8x!ttcore.tile<32x32, f32>, #l1_>)
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}

func.func @matmul_multi_core_dram_params(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #dram>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  %stream = "d2m.stream_layout"(%arg0, %alloc_0) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  %stream_2 = "d2m.stream_layout"(%arg1, %alloc_1) : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #dram>, memref<2x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>
  "d2m.generic"(%stream, %stream_2, %alloc) <{block_factors = [1, 1, 4], grid = #ttcore.grid<2x4>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %core0 = d2m.core_index(0) : index
    %core1 = d2m.core_index(1) : index
    %0 = arith.cmpi eq, %core1, %c0 : index
    scf.if %0 {
      // CHECK-DAG: [[iter0:%.*]] = d2m.iter_index(0)
      // CHECK-DAG: [[iter2:%.*]] = d2m.iter_index(2)
      // CHECK-DAG: [[core0:%.*]] = d2m.core_index(0)
      // CHECK-DAG: [[offset:%.*]] = arith.addi [[core0]], [[iter0]]
      // CHECK: d2m.dma_read %stream{{[_0-9]*}}[[[offset]], [[iter2]], %c0]
      %tx = d2m.dma %stream<#map1>, %cb0 : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      d2m.dma_wait %tx
      d2m.semaphore_wait %sem0, %c3 reset %c0
      %tx_3 = d2m.dma %cb0, %cb0 core[%core0, %c0] mcast[%c1, %c4] : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      d2m.dma_wait %tx_3
      d2m.semaphore_set %sem1, %c1, core[%core0, %c0] mcast[%c1, %c4]
    } else {
      d2m.semaphore_inc %sem0, %c1, core[%core0, %c0]
      d2m.semaphore_wait %sem1, %c1 reset %c0
    }
    d2m.yield %cb0 : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %core0 = d2m.core_index(0) : index
    %0 = arith.cmpi eq, %core0, %c0 : index
    %core1 = d2m.core_index(1) : index
    scf.if %0 {
      // CHECK-DAG: [[iter2:%.*]] = d2m.iter_index(2)
      // CHECK-DAG: [[core1:%.*]] = d2m.core_index(1)
      // CHECK: d2m.dma_read %stream{{[_0-9]*}}
      %tx = d2m.dma %stream_2<#map2>, %cb1 : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      d2m.dma_wait %tx
      d2m.semaphore_wait %sem2, %c1 reset %c0
      %tx_3 = d2m.dma %cb1, %cb1 core[%c0, %core1] mcast[%c2, %c1] : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>) -> !d2m.mem_tx
      d2m.dma_wait %tx_3
      d2m.semaphore_set %sem3, %c1, core[%c0, %core1] mcast[%c2, %c1]
    } else {
      d2m.semaphore_inc %sem2, %c1, core[%c0, %core1]
      d2m.semaphore_wait %sem3, %c1 reset %c0
    }
    d2m.yield %cb1 : (memref<6x8x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement2(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    d2m.await %cb2 : (memref<4x8x!ttcore.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    d2m.await %cb0, %cb1 : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>)
    "d2m.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
    d2m.yield %cb2 : (memref<4x8x!ttcore.tile<32x32, f32>, #l1_>)
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}
