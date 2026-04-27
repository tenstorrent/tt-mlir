// RUN: ttmlir-opt --ttcore-register-device --d2m-expand-dma-read-composite-view %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
module attributes {} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

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
      %1 = d2m.get_cb(1) : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
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
      %1 = d2m.get_cb(1) : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
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

  // CHECK-LABEL: func.func @test_composite_view_row_major_width
  func.func @test_composite_view_row_major_width() -> memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 16, 16>}> : (memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>, memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>) -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<4x1x32x32xf32, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>)
     {
      %1 = d2m.get_cb(1) operand_index = 1 resolution_stage =  compile : <memref<32x32xf32, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      // CHECK: d2m.reserve
      %2 = d2m.reserve %1 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
      // CHECK: scf.for
      // CHECK: scf.for
      // CHECK: scf.if
      // CHECK-NOT: scf.if
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      // CHECK: else
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<4x1x32x32xf32, #ttcore.view<4>, #l1>, memref<32x32xf32, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      // CHECK: d2m.push
      d2m.push %1 : <memref<32x32xf32, #l1>>
    }
    memref.dealloc %alloc_1 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    memref.dealloc %alloc_0 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    return %alloc_2 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
  }

  // CHECK-LABEL: func.func @test_composite_view_row_major_height
  func.func @test_composite_view_row_major_height() -> memref<2x16x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64, d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3, d4, d5) -> (((d1 + d2) floordiv 7 + d0) mod 2, (d1 + d2) mod 7, d3, d4, d5)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, (d1 floordiv 7 + d0) mod 2, d1 mod 7, 0)>} : memref<2x7x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64, d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3, d4, d5) -> (((d1 + d2) floordiv 7 + d0) mod 2, (d1 + d2) mod 7, d3, d4, d5)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, (d1 floordiv 7 + d0) mod 2, d1 mod 7, 0)>} : memref<2x7x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 223, 223>}> : (memref<2x7x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>, memref<2x7x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>) -> memref<2x16x1x1x32x32xf32, #ttcore.view<6>, #l1>
    %alloc_2 = memref.alloc() {address = 120096 : i64, alignment = 16 : i64, d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3, d4, d5) -> (((d1 + d2) floordiv 8 + d0 * 2) mod 4, (d1 + d2) mod 8, d3, d4, d5)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, ((d1 + d0 * 8) floordiv 16) mod 2, (d1 + d0 * 8) mod 16, 0)>} : memref<2x16x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x16x1, virt_to_physical_map = (d0, d1, d2) -> (0, ((d1 + d2) floordiv 8 + d0 * 2) mod 4, (d1 + d2) mod 8), physical_to_virt_map = (d0, d1) -> (0, ((d1 + d0 * 8) floordiv 16) mod 2, (d1 + d0 * 8) mod 16, 0)>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<2x16x1x1x32x32xf32, #ttcore.view<6>, #l1>)
        outs(%alloc_2 : memref<2x16x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>)
     {
      %1 = d2m.get_cb(1) operand_index = 1 resolution_stage =  compile : <memref<1x32x32xf32, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<(d0, d1) -> (0, ((d1 + d0 * 8) floordiv 16) mod 2, (d1 + d0 * 8) mod 16, 0)>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<(d0, d1) -> (0, ((d1 + d0 * 8) floordiv 16) mod 2, (d1 + d0 * 8) mod 16, 0)>} : index
      %core2 = d2m.core_index(2) {phys_to_virt_map = affine_map<(d0, d1) -> (0, ((d1 + d0 * 8) floordiv 16) mod 2, (d1 + d0 * 8) mod 16, 0)>} : index
      // CHECK: d2m.reserve
      %2 = d2m.reserve %1 : <memref<1x32x32xf32, #l1>> -> memref<1x32x32xf32, #l1>
      // CHECK: scf.for
      // CHECK: scf.for
      // CHECK: scf.for
      // CHECK: scf.if
      // CHECK: scf.if
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      // CHECK: else
      // CHECK: d2m.dma_read
      // CHECK: d2m.dma_wait
      %tx = d2m.dma_read %0[%core0, %core1, %core2], %2, <0> : (memref<2x16x1x1x32x32xf32, #ttcore.view<6>, #l1>, memref<1x32x32xf32, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      // CHECK: d2m.push
      d2m.push %1 : <memref<1x32x32xf32, #l1>>
    }
    memref.dealloc %alloc_0 : memref<2x7x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>
    memref.dealloc %alloc_1 : memref<2x7x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>
    return %alloc_2 : memref<2x16x1x1x32x32xf32, #ttcore.shard<4096x128x4, 1>, #l1>
  }
}
