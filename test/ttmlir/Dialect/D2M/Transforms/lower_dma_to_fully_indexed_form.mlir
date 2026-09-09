// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-dma-to-fully-indexed-form %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-dma-to-fully-indexed-form="use-tensor-accessor-dma=true" %s | FileCheck %s --check-prefix=CHECK-TA
// CHECK-TA-DAG: #[[REORDER_PAGE_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 * 8 + d1 + d2 * 4 + d3 * 2)>

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#reorder = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
#parallel = #ttcore.iterator_type<parallel>
module attributes {} {
  // CHECK-LABEL: func.func @test_shard_level_read
  // CHECK-TA-LABEL: func.func @test_shard_level_read
  func.func @test_shard_level_read(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
          %local = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          // CHECK-NOT: d2m.dma_read {{.*}}, <0>
          // CHECK: d2m.dma_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}[%{{.*}}], <8>
          // CHECK-TA: d2m.dma_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}, <0> {tensorAccessorPageMap = #{{.*}}}
          %tx = d2m.dma_read %stream[%0, %1], %local, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx : !d2m.mem_tx<read>
          d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }
    return
  }

  // CHECK-LABEL: func.func @test_shard_level_write
  // CHECK-TA-LABEL: func.func @test_shard_level_write
  func.func @test_shard_level_write(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          %local = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          // CHECK-NOT: d2m.dma_write {{.*}}, <0>
          // CHECK: d2m.dma_write %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], <8>
          // CHECK-TA: d2m.dma_write %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}], <0> {tensorAccessorPageMap = #{{.*}}}
          %tx = d2m.dma_write %local, %stream[%0, %1], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx<write>
          d2m.dma_wait %tx : !d2m.mem_tx<write>
          d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }
    return
  }

  // CHECK-LABEL: func.func @test_already_fully_indexed
  // CHECK-TA-LABEL: func.func @test_already_fully_indexed
  func.func @test_already_fully_indexed(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %local = d2m.reserve %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          // CHECK: d2m.dma_read %view[%core0, %core1, %c0], %{{.*}}[%c0], <8>
          // CHECK-TA: d2m.dma_read %view[%core0, %core1, %c0], %{{.*}}[%c0], <8>
          %tx = d2m.dma_read %stream[%core0, %core1, %c0], %local[%c0], <8> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
          d2m.dma_wait %tx : !d2m.mem_tx<read>
          d2m.push %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }
    return
  }

  // Test: Identity local_copy with contiguous layout - fully coalesced single dma_read.
  // CHECK-LABEL: func.func @test_local_copy_identity
  // CHECK-TA-LABEL: func.func @test_local_copy_identity
  func.func @test_local_copy_identity() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    // CHECK-NOT: d2m.local_copy
    // CHECK: d2m.wait
    // CHECK: d2m.reserve
    // CHECK: d2m.dma_read %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], <32>
    // CHECK: d2m.dma_wait
    // CHECK: d2m.push
    // CHECK: d2m.pop
    // CHECK: }, {
    // CHECK-TA-NOT: d2m.local_copy
    // CHECK-TA: d2m.dma_read %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], <32>
    ^datamovement0:
      %cb_src = d2m.get_cb(0) : !d2m.cb<memref<4x8xbf16, #l1>>
      %cb_dst = d2m.get_cb(3) : !d2m.cb<memref<4x8xbf16, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %src = d2m.wait %cb_src : <memref<4x8xbf16, #l1>> -> memref<4x8xbf16, #l1>
          %dst = d2m.reserve %cb_dst : <memref<4x8xbf16, #l1>> -> memref<4x8xbf16, #l1>
          %tx = d2m.local_copy %src, %dst indexing_maps = [#map, #map] : memref<4x8xbf16, #l1>, memref<4x8xbf16, #l1> -> !d2m.mem_tx<read>
          d2m.dma_wait %tx : !d2m.mem_tx<read>
          d2m.push %cb_dst : <memref<4x8xbf16, #l1>>
          d2m.pop %cb_src : <memref<4x8xbf16, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb_dst = d2m.get_cb(3) : !d2m.cb<memref<4x8xbf16, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %result = d2m.wait %cb_dst : <memref<4x8xbf16, #l1>> -> memref<4x8xbf16, #l1>
          d2m.pop %cb_dst : <memref<4x8xbf16, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test: Transposed src map breaks contiguity -> per-element DMA with coalescing=1.
  // CHECK-LABEL: func.func @test_local_copy_transposed_src
  func.func @test_local_copy_transposed_src() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    // CHECK-NOT: d2m.local_copy
    // CHECK: d2m.wait
    // CHECK: d2m.reserve
    // CHECK: d2m.dma_read %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}], <1>
    // CHECK: d2m.dma_wait
    // CHECK: d2m.push
    // CHECK: d2m.pop
    // CHECK: }, {
    ^datamovement0:
      %cb_src = d2m.get_cb(0) : !d2m.cb<memref<8x4xbf16, #l1>>
      %cb_dst = d2m.get_cb(3) : !d2m.cb<memref<4x8xbf16, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %src = d2m.wait %cb_src : <memref<8x4xbf16, #l1>> -> memref<8x4xbf16, #l1>
          %dst = d2m.reserve %cb_dst : <memref<4x8xbf16, #l1>> -> memref<4x8xbf16, #l1>
          %tx = d2m.local_copy %src, %dst indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>] : memref<8x4xbf16, #l1>, memref<4x8xbf16, #l1> -> !d2m.mem_tx<read>
          d2m.dma_wait %tx : !d2m.mem_tx<read>
          d2m.push %cb_dst : <memref<4x8xbf16, #l1>>
          d2m.pop %cb_src : <memref<8x4xbf16, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb_dst = d2m.get_cb(3) : !d2m.cb<memref<4x8xbf16, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %result = d2m.wait %cb_dst : <memref<4x8xbf16, #l1>> -> memref<4x8xbf16, #l1>
          d2m.pop %cb_dst : <memref<4x8xbf16, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // CHECK-TA-LABEL: func.func @test_row_major_fallback
  // CHECK-TA-NOT: tensorAccessorPageMap
  // CHECK-TA-NOT: d2m.dma_write {{.*}}, <0>
  // CHECK-TA: d2m.dma_write
  func.func @test_row_major_fallback(
      %remote: memref<1x1x4x4xf32, #ttcore.shard<16x4, 1>, #dram>,
      %local: memref<4x4xf32, #l1>) {
    %c0 = arith.constant 0 : index
    %tx = d2m.dma_write %local, %remote[%c0, %c0], <0> : (memref<4x4xf32, #l1>, memref<1x1x4x4xf32, #ttcore.shard<16x4, 1>, #dram>) -> !d2m.mem_tx<write>
    return
  }

  // CHECK-TA-LABEL: func.func @test_mcast_fallback
  // CHECK-TA-NOT: tensorAccessorPageMap
  // CHECK-TA: d2m.dma_write %{{.*}}[%{{.*}}], %{{.*}}[%{{.*}}] core
  func.func @test_mcast_fallback(%local: memref<4x4xf32, #l1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %tx = d2m.dma_write %local, %local core[%c0, %c0] mcast[%c1, %c2], <0> : (memref<4x4xf32, #l1>, memref<4x4xf32, #l1>) -> !d2m.mem_tx<mcast_write>
    return
  }

  // CHECK-TA-LABEL: func.func @test_fabric_fallback
  // CHECK-TA-NOT: tensorAccessorPageMap
  // CHECK-TA-NOT: d2m.dma_write {{.*}}, <0>
  // CHECK-TA: d2m.dma_write
  func.func @test_fabric_fallback(
      %remote: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>,
      %local: memref<2x2x!ttcore.tile<32x32, f32>, #l1>) {
    %c0 = arith.constant 0 : index
    %tx = d2m.dma_write %local, %remote[%c0, %c0] startDevice[%c0, %c0], <0> : (memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>) -> !d2m.mem_tx<write>
    return
  }

  // CHECK-TA-LABEL: func.func @test_non_identity_view_page_map
  // CHECK-TA: d2m.dma_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}, <0> {tensorAccessorPageMap = #[[REORDER_PAGE_MAP]]}
  func.func @test_non_identity_view_page_map(%remote: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    %view = d2m.view_layout %remote remapping = #reorder : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram> -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%view : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    ^datamovement0:
      %cb = d2m.get_cb(0) : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>
      %local = d2m.reserve %cb : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %tx = d2m.dma_read %view[%c0, %c1], %local, <0> : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
    }, {
    ^compute0:
    }
    return
  }

  // CHECK-TA-LABEL: func.func @test_in_thread_view_fallback
  // CHECK-TA-NOT: tensorAccessorPageMap
  // CHECK-TA-NOT: d2m.dma_read {{.*}}, <0>
  // CHECK-TA: d2m.dma_read
  func.func @test_in_thread_view_fallback(%remote: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%remote : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram>)
        outs(%alloc : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    ^datamovement0:
      %view = d2m.view_layout %remote remapping = #reorder : memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #dram> -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
      %cb = d2m.get_cb(0) : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>
      %local = d2m.reserve %cb : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %tx = d2m.dma_read %view[%c0, %c1], %local, <0> : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
    }, {
    ^compute0:
    }
    return
  }
}
