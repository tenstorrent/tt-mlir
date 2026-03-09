// RUN: ttmlir-opt --ttcore-register-device --d2m-optimize-dma %s | FileCheck %s

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>

module attributes {} {
  // Two shard-level reads into different CBs.
  // CHECK-LABEL: func.func @test_coalesce_two_reads
  func.func @test_coalesce_two_reads(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc1 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_out = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      // CHECK: d2m.reserve %cb0
      // CHECK: [[TX0:%.+]] = d2m.dma_read
      // CHECK: d2m.reserve %cb1
      // CHECK: [[TX1:%.+]] = d2m.dma_read
      // CHECK-NEXT: d2m.dma_wait [[TX0]] : !d2m.mem_tx<read>
      // CHECK-NEXT: d2m.push %cb0
      // CHECK-NEXT: d2m.dma_wait [[TX1]] : !d2m.mem_tx<read>
      // CHECK-NEXT: d2m.push %cb1
      %local0 = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx0 = d2m.dma_read %stream0[%gi, %gj], %local0, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx0 : !d2m.mem_tx<read>
      d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %local1 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx1 = d2m.dma_read %stream1[%gi, %gj], %local1, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx1 : !d2m.mem_tx<read>
      d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Three shard-level reads into different CBs — 3-wide coalescing.
  // CHECK-LABEL: func.func @test_coalesce_three_reads
  func.func @test_coalesce_three_reads(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg2: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc1 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc2 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_out = memref.alloc() {address = 13312 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream2 = "d2m.stream_layout"(%arg2, %alloc2) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0, %stream1, %stream2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      // CHECK: d2m.reserve %cb0
      // CHECK: [[TX0:%.+]] = d2m.dma_read
      // CHECK: d2m.reserve %cb1
      // CHECK: [[TX1:%.+]] = d2m.dma_read
      // CHECK: d2m.reserve %cb2
      // CHECK: [[TX2:%.+]] = d2m.dma_read
      // CHECK-NEXT: d2m.dma_wait [[TX0]] : !d2m.mem_tx<read>
      // CHECK-NEXT: d2m.push %cb0
      // CHECK-NEXT: d2m.dma_wait [[TX1]] : !d2m.mem_tx<read>
      // CHECK-NEXT: d2m.push %cb1
      // CHECK-NEXT: d2m.dma_wait [[TX2]] : !d2m.mem_tx<read>
      // CHECK-NEXT: d2m.push %cb2
      %local0 = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx0 = d2m.dma_read %stream0[%gi, %gj], %local0, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx0 : !d2m.mem_tx<read>
      d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %local1 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx1 = d2m.dma_read %stream1[%gi, %gj], %local1, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx1 : !d2m.mem_tx<read>
      d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %local2 = d2m.reserve %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx2 = d2m.dma_read %stream2[%gi, %gj], %local2, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx2 : !d2m.mem_tx<read>
      d2m.push %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Single shard-level read: no coalescing, order unchanged.
  // CHECK-LABEL: func.func @test_single_read_no_change
  func.func @test_single_read_no_change(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %alloc_0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      // CHECK: d2m.reserve %cb0
      // CHECK: [[TX:%.+]] = d2m.dma_read
      // CHECK-NEXT: d2m.dma_wait [[TX]] : !d2m.mem_tx<read>
      // CHECK-NEXT: d2m.push %cb0
      %local = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx = d2m.dma_read %stream[%gi, %gj], %local, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Read + write (different CBs): both DMA ops issued before either barrier.
  // CHECK-LABEL: func.func @test_read_and_write
  func.func @test_read_and_write(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream_1 = "d2m.stream_layout"(%arg1, %alloc_0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream_1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      // CHECK: d2m.reserve %cb0
      // CHECK: [[TX_R:%.+]] = d2m.dma_read
      // CHECK: d2m.wait %cb1
      // CHECK: [[TX_W:%.+]] = d2m.dma_write
      // CHECK-NEXT: d2m.dma_wait [[TX_R]] : !d2m.mem_tx<read>
      // CHECK-NEXT: d2m.push %cb0
      // CHECK-NEXT: d2m.dma_wait [[TX_W]] : !d2m.mem_tx<write>
      // CHECK-NEXT: d2m.pop %cb1
      %local0 = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx0 = d2m.dma_read %stream[%gi, %gj], %local0, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx0 : !d2m.mem_tx<read>
      d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %local1 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx1 = d2m.dma_write %local1, %stream_1[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
      d2m.dma_wait %tx1 : !d2m.mem_tx<write>
      d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Two shard-level writes from different CBs.
  // CHECK-LABEL: func.func @test_coalesce_two_writes
  func.func @test_coalesce_two_writes(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc1 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_out = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      // CHECK: d2m.wait %cb0
      // CHECK: [[TX0:%.+]] = d2m.dma_write
      // CHECK: d2m.wait %cb1
      // CHECK: [[TX1:%.+]] = d2m.dma_write
      // CHECK-NEXT: d2m.dma_wait [[TX0]] : !d2m.mem_tx<write>
      // CHECK-NEXT: d2m.pop %cb0
      // CHECK-NEXT: d2m.dma_wait [[TX1]] : !d2m.mem_tx<write>
      // CHECK-NEXT: d2m.pop %cb1
      %local0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx0 = d2m.dma_write %local0, %stream0[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
      d2m.dma_wait %tx0 : !d2m.mem_tx<write>
      d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %local1 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %tx1 = d2m.dma_write %local1, %stream1[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
      d2m.dma_wait %tx1 : !d2m.mem_tx<write>
      d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Read + write in a loop (different CBs.
  // CHECK-LABEL: func.func @test_defer_write_barrier
  func.func @test_defer_write_barrier(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc1 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // CHECK: [[NULL:%.+]] = d2m.null_tx : !d2m.mem_tx<write>
      // CHECK: [[FOR:%.+]] = scf.for {{.+}} iter_args([[PREV:%.+]] = [[NULL]])
      // CHECK:   d2m.reserve %cb0
      // CHECK:   [[TX_R:%.+]] = d2m.dma_read
      // CHECK:   scf.if
      // CHECK:     d2m.dma_wait [[PREV]] : !d2m.mem_tx<write>
      // CHECK:     d2m.pop %cb1
      // CHECK:   d2m.wait %cb1
      // CHECK:   [[TX_W:%.+]] = d2m.dma_write
      // CHECK:   d2m.dma_wait [[TX_R]] : !d2m.mem_tx<read>
      // CHECK:   d2m.push %cb0
      // CHECK:   scf.yield [[TX_W]]
      // CHECK: d2m.dma_wait [[FOR]] : !d2m.mem_tx<write>
      // CHECK-NEXT: d2m.pop %cb1
      scf.for %iv = %c0 to %c8 step %c1 {
        %local_in = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_r = d2m.dma_read %stream0[%gi, %gj], %local_in, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx_r : !d2m.mem_tx<read>
        d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        %local_out = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_w = d2m.dma_write %local_out, %stream1[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx_w : !d2m.mem_tx<write>
        d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Two reads + one write in a loop (all different CBs).
  // CHECK-LABEL: func.func @test_defer_write_with_coalesced_reads
  func.func @test_defer_write_with_coalesced_reads(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg2: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc1 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc2 = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream2 = "d2m.stream_layout"(%arg2, %alloc2) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // CHECK: [[NULL:%.+]] = d2m.null_tx : !d2m.mem_tx<write>
      // CHECK: [[FOR:%.+]] = scf.for {{.+}} iter_args([[PREV:%.+]] = [[NULL]])
      // CHECK:   d2m.reserve %cb0
      // CHECK:   [[TX_R0:%.+]] = d2m.dma_read
      // CHECK:   d2m.reserve %cb1
      // CHECK:   [[TX_R1:%.+]] = d2m.dma_read
      // CHECK:   scf.if
      // CHECK:     d2m.dma_wait [[PREV]] : !d2m.mem_tx<write>
      // CHECK:     d2m.pop %cb2
      // CHECK:   d2m.wait %cb2
      // CHECK:   [[TX_W:%.+]] = d2m.dma_write
      // CHECK:   d2m.dma_wait [[TX_R0]] : !d2m.mem_tx<read>
      // CHECK:   d2m.push %cb0
      // CHECK:   d2m.dma_wait [[TX_R1]] : !d2m.mem_tx<read>
      // CHECK:   d2m.push %cb1
      // CHECK:   scf.yield [[TX_W]]
      // CHECK: d2m.dma_wait [[FOR]] : !d2m.mem_tx<write>
      // CHECK-NEXT: d2m.pop %cb2
      scf.for %iv = %c0 to %c8 step %c1 {
        %local0 = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_r0 = d2m.dma_read %stream0[%gi, %gj], %local0, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx_r0 : !d2m.mem_tx<read>
        d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        %local1 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_r1 = d2m.dma_read %stream1[%gi, %gj], %local1, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx_r1 : !d2m.mem_tx<read>
        d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        %local_out = d2m.wait %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_w = d2m.dma_write %local_out, %stream2[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx_w : !d2m.mem_tx<write>
        d2m.pop %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Write-only loop (no reads) — write barrier should still be deferred.
  // CHECK-LABEL: func.func @test_defer_write_only_loop
  func.func @test_defer_write_only_loop(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins() outs(%stream0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // CHECK: [[NULL:%.+]] = d2m.null_tx : !d2m.mem_tx<write>
      // CHECK: [[FOR:%.+]] = scf.for {{.+}} iter_args([[PREV:%.+]] = [[NULL]])
      // CHECK:   scf.if
      // CHECK:     d2m.dma_wait [[PREV]] : !d2m.mem_tx<write>
      // CHECK:     d2m.pop %cb0
      // CHECK:   d2m.wait %cb0
      // CHECK:   [[TX_W:%.+]] = d2m.dma_write
      // CHECK:   scf.yield [[TX_W]]
      // CHECK: d2m.dma_wait [[FOR]] : !d2m.mem_tx<write>
      // CHECK-NEXT: d2m.pop %cb0
      scf.for %iv = %c0 to %c8 step %c1 {
        %local_out = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_w = d2m.dma_write %local_out, %stream0[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx_w : !d2m.mem_tx<write>
        d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Read + write using the SAME CB in a loop — aliased CBs prevent deferral.
  // CHECK-LABEL: func.func @test_no_defer_aliased_cbs
  func.func @test_no_defer_aliased_cbs(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc1 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // Both read and write use cb0 — aliased, no deferral.
      // sinkBarriers won't move barriers here since read/write use same CB.
      // CHECK-NOT: d2m.null_tx
      // CHECK: scf.for
      // CHECK:   [[TX_R:%.+]] = d2m.dma_read
      // CHECK:   d2m.dma_wait [[TX_R]] : !d2m.mem_tx<read>
      // CHECK:   [[TX_W:%.+]] = d2m.dma_write
      // CHECK:   d2m.dma_wait [[TX_W]] : !d2m.mem_tx<write>
      // CHECK:   d2m.pop %cb0
      scf.for %iv = %c0 to %c8 step %c1 {
        %local_in = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_r = d2m.dma_read %stream0[%gi, %gj], %local_in, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx_r : !d2m.mem_tx<read>
        d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        %local_out = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_w = d2m.dma_write %local_out, %stream1[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx_w : !d2m.mem_tx<write>
        d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }

  // Read + mcast write in a loop (different CBs).
  // CHECK-LABEL: func.func @test_defer_mcast_write_barrier
  func.func @test_defer_mcast_write_barrier(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc1 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %sem0: !d2m.semaphore):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // CHECK: [[NULL:%.+]] = d2m.null_tx : !d2m.mem_tx<mcast_write>
      // CHECK: [[FOR:%.+]] = scf.for {{.+}} iter_args([[PREV:%.+]] = [[NULL]])
      // CHECK:   d2m.reserve %cb0
      // CHECK:   [[TX_R:%.+]] = d2m.dma_read
      // CHECK:   scf.if
      // CHECK:     d2m.dma_wait [[PREV]] : !d2m.mem_tx<mcast_write>
      // CHECK:     d2m.semaphore_set
      // CHECK:     d2m.pop %cb1
      // CHECK:   d2m.wait %cb1
      // CHECK:   d2m.dma_wait [[TX_R]] : !d2m.mem_tx<read>
      // CHECK:   d2m.push %cb0
      // CHECK:   d2m.semaphore_wait
      // CHECK:   [[TX_W:%.+]] = d2m.dma_write {{.*}} mcast
      // CHECK:   scf.yield [[TX_W]]
      // CHECK: d2m.dma_wait [[FOR]] : !d2m.mem_tx<mcast_write>
      // CHECK-NEXT: d2m.semaphore_set
      // CHECK-NEXT: d2m.pop %cb1
      scf.for %iv = %c0 to %c8 step %c1 {
        %local_in = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_r = d2m.dma_read %stream0[%gi, %gj], %local_in, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
        d2m.dma_wait %tx_r : !d2m.mem_tx<read>
        d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        %local_out = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        d2m.semaphore_wait %sem0, %c1 : !d2m.semaphore
        %tx_w = d2m.dma_write %local_out, %local_out core[%gi, %gj] mcast[%c8, %c1], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<mcast_write>
        d2m.dma_wait %tx_w : !d2m.mem_tx<mcast_write>
        d2m.semaphore_set %sem0, %c0 : !d2m.semaphore
        d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %sem0: !d2m.semaphore):
    }
    return
  }

  // Two writes in a loop using different CBs (no reads). Both write barriers are deferred.
  // CHECK-LABEL: func.func @test_defer_two_writes
  func.func @test_defer_two_writes(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc1 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_out = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %gi = d2m.core_index(0) : index
      %gj = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // CHECK: [[NULL0:%.+]] = d2m.null_tx : !d2m.mem_tx<write>
      // CHECK: [[NULL1:%.+]] = d2m.null_tx : !d2m.mem_tx<write>
      // CHECK: [[FOR:%.+]]:2 = scf.for {{.+}} iter_args([[PREV0:%.+]] = [[NULL0]], [[PREV1:%.+]] = [[NULL1]])
      // CHECK:   scf.if
      // CHECK:     d2m.dma_wait [[PREV1]] : !d2m.mem_tx<write>
      // CHECK:     d2m.pop %cb0
      // CHECK:   d2m.wait %cb0
      // CHECK:   [[TX0:%.+]] = d2m.dma_write
      // CHECK:   scf.if
      // CHECK:     d2m.dma_wait [[PREV0]] : !d2m.mem_tx<write>
      // CHECK:     d2m.pop %cb1
      // CHECK:   d2m.wait %cb1
      // CHECK:   [[TX1:%.+]] = d2m.dma_write
      // CHECK:   scf.yield [[TX1]], [[TX0]]
      // CHECK: d2m.dma_wait [[FOR]]#1 : !d2m.mem_tx<write>
      // CHECK-NEXT: d2m.pop %cb0
      // CHECK: d2m.dma_wait [[FOR]]#0 : !d2m.mem_tx<write>
      // CHECK-NEXT: d2m.pop %cb1
      scf.for %iv = %c0 to %c8 step %c1 {
        %local_out0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_w0 = d2m.dma_write %local_out0, %stream0[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx_w0 : !d2m.mem_tx<write>
        d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        %local_out1 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %tx_w1 = d2m.dma_write %local_out1, %stream1[%gi, %gj], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
        d2m.dma_wait %tx_w1 : !d2m.mem_tx<write>
        d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      }
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }
}
