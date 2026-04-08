// RUN: ttmlir-opt --ttcore-register-device --d2m-expand-dma-read-composite-view %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
module attributes {} {
  // CHECK-LABEL: func.func @test_composite_view_small
  func.func @test_composite_view_small() -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^datamovement0:
      %1 = d2m.get_cb(1) operand_index = 1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      // CHECK: d2m.reserve
      %2 = d2m.reserve %1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: scf.if
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      // CHECK: else
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      // CHECK: d2m.push
      d2m.push %1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    }
    memref.dealloc %alloc_0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    memref.dealloc %alloc_1 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    return %alloc_2 : memref<1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }

  // CHECK-LABEL: func.func @test_composite_view_large_grid_padded
  func.func @test_composite_view_large_grid_padded() -> memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 132384 : i64, alignment = 16 : i64} : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1>
    %view_0 = d2m.view_layout %alloc_0 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 mod 7)> : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1> -> memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %view_1 = d2m.view_layout %alloc_1 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 mod 7)> : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1> -> memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%view_0, %view_1) <{dim = 1 : si32}> : (memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>) -> memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 161056 : i64, alignment = 16 : i64} : memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) {
    ^datamovement0:
      %1 = d2m.get_cb(1) operand_index = 1 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      // CHECK: d2m.reserve
      %2 = d2m.reserve %1 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x2x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: scf.if
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      // CHECK: else
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x2x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      // CHECK: d2m.push
      d2m.push %1 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
    }
    memref.dealloc %alloc_0 : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1>
    memref.dealloc %alloc_1 : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1>
    return %alloc_2 : memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
  }
}
