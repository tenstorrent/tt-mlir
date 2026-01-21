// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-dma-to-fully-indexed-form %s | FileCheck %s

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
module attributes {} {
  // CHECK-LABEL: func.func @test_shard_level_read
  func.func @test_shard_level_read(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %alloc_0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
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
          %tx = d2m.dma_read %stream[%0, %1], %local, <0> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
          d2m.dma_wait %tx
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
  func.func @test_shard_level_write(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg1, %alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
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
          %tx = d2m.dma_write %local, %stream[%0, %1], <0> : (memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) -> !d2m.mem_tx
          d2m.dma_wait %tx
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
  func.func @test_already_fully_indexed(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %alloc_0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
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
          // CHECK: d2m.dma_read %stream[%core0, %core1, %c0], %{{.*}}[%c0], <8>
          %tx = d2m.dma_read %stream[%core0, %core1, %c0], %local[%c0], <8> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
          d2m.dma_wait %tx
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

  // CHECK-LABEL: func.func @test_composite_view_small
  func.func @test_composite_view_small() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    %alloc_2 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>)
        outs(%alloc_2 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>) {
    ^datamovement0:
      %1 = d2m.get_cb(1) operand_index = 1 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      // CHECK: d2m.reserve
      %2 = d2m.reserve %1 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      // CHECK: scf.if
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      // CHECK: else
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) -> !d2m.mem_tx
      d2m.dma_wait %tx
      // CHECK: d2m.push
      d2m.push %1 : <memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>
    }
    memref.dealloc %alloc_0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    memref.dealloc %alloc_1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    return %alloc_2 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
  }

  // CHECK-LABEL: func.func @test_composite_view_large_grid_padded
  func.func @test_composite_view_large_grid_padded() -> memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_1 = memref.alloc() {address = 132384 : i64, alignment = 16 : i64} : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #ttcore.memory_space<l1>>
    %view_0 = d2m.view_layout %alloc_0 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 mod 7)> : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    %view_1 = d2m.view_layout %alloc_1 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 mod 7)> : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%view_0, %view_1) <{dim = 1 : si32}> : (memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>, memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>) -> memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    %alloc_2 = memref.alloc() {address = 161056 : i64, alignment = 16 : i64} : memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>)
        outs(%alloc_2 : memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>>) {
    ^datamovement0:
      %1 = d2m.get_cb(1) operand_index = 1 : <memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      // CHECK: d2m.reserve
      %2 = d2m.reserve %1 : <memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      // CHECK: scf.if
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      // CHECK: else
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<l1>>, memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) -> !d2m.mem_tx
      d2m.dma_wait %tx
      // CHECK: d2m.push
      d2m.push %1 : <memref<1x2x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>
    }
    memref.dealloc %alloc_0 : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #ttcore.memory_space<l1>>
    memref.dealloc %alloc_1 : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #ttcore.memory_space<l1>>
    return %alloc_2 : memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #ttcore.memory_space<l1>>
  }
}
