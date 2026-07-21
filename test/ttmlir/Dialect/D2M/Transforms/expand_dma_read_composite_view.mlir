// RUN: ttmlir-opt --ttcore-register-device --d2m-expand-dma-read-composite-view %s | FileCheck %s
// RUN: ttmlir-opt --ttcore-register-device --d2m-expand-dma-read-composite-view %s | FileCheck %s --check-prefix=COUNT

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

  // CHECK-LABEL: func.func @test_view_layout_of_composite_view
  func.func @test_view_layout_of_composite_view() -> memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<1x1x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    // CHECK-NOT: d2m.composite_view
    // CHECK-NOT: d2m.view_layout
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32}> : (memref<1x1x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %view = d2m.view_layout %0 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 * 2 + d3)> : memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 120096 : i64, alignment = 16 : i64} : memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%view : memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^datamovement0:
      // CHECK: d2m.get_cb(2)
      // CHECK-NOT: d2m.composite_view
      // CHECK-NOT: d2m.view_layout
      %1 = d2m.get_cb(1) : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %2 = d2m.reserve %1 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x2x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: d2m.dma_read
      // CHECK-NOT: d2m.composite_view
      // CHECK-NOT: d2m.view_layout
      %tx = d2m.dma_read %view[%core0, %core1], %2, <0> : (memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x2x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      d2m.push %1 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
    }
    // CHECK: return
    return %alloc_2 : memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }

  // CHECK-LABEL: func.func @test_view_layout_chain_of_composite_view
  func.func @test_view_layout_chain_of_composite_view() -> memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<1x1x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    // CHECK-NOT: d2m.composite_view
    // CHECK-NOT: d2m.view_layout
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32}> : (memref<1x1x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %view_0 = d2m.view_layout %0 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %view_1 = d2m.view_layout %view_0 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 * 2 + d3)> : memref<1x1x1x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 120096 : i64, alignment = 16 : i64} : memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%view_1 : memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^datamovement0:
      // CHECK: d2m.get_cb(2)
      // CHECK-NOT: d2m.composite_view
      // CHECK-NOT: d2m.view_layout
      %1 = d2m.get_cb(1) : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %2 = d2m.reserve %1 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x2x!ttcore.tile<32x32, f32>, #l1>
      // CHECK: d2m.dma_read
      // CHECK-NOT: d2m.composite_view
      // CHECK-NOT: d2m.view_layout
      %tx = d2m.dma_read %view_1[%core0, %core1], %2, <0> : (memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x2x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      d2m.push %1 : <memref<1x2x!ttcore.tile<32x32, f32>, #l1>>
    }
    // CHECK: return
    return %alloc_2 : memref<1x2x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
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
      %1 = d2m.get_cb(1) resolution_stage =  compile : <memref<32x32xf32, #l1>>
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
      %1 = d2m.get_cb(1) resolution_stage =  compile : <memref<1x32x32xf32, #l1>>
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

  // Tiled chained-views variant of @test_composite_view_large_grid_padded:
  // each input fed by a 2-step view_layout chain.
  // CHECK-LABEL: func.func @test_composite_view_chained_views_tiled
  func.func @test_composite_view_chained_views_tiled() -> memref<1x8x1x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 132384 : i64, alignment = 16 : i64} : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1>
    %pre_0 = d2m.view_layout %alloc_0 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1> -> memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %pre_1 = d2m.view_layout %alloc_1 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.shard<28672x4096, 1>, #l1> -> memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %view_0 = d2m.view_layout %pre_0 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 mod 7)> : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %view_1 = d2m.view_layout %pre_1 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d1 mod 7)> : memref<1x1x1x7x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1> -> memref<1x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
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

  // Row-major chained-views variant with mixed chain depths (1, 2) per input.
  // CHECK-LABEL: func.func @test_composite_view_chained_views_row_major
  func.func @test_composite_view_chained_views_row_major() -> memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %pre_1 = d2m.view_layout %alloc_1 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1> -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1>
    %view_0 = d2m.view_layout %alloc_0 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1> -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1>
    %view_1 = d2m.view_layout %pre_1 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> : memref<4x1x32x32xf32, #ttcore.view<4>, #l1> -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1>
    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%view_0, %view_1) <{dim = 1 : si32, logicalSizes = array<i64: 16, 16>}> : (memref<4x1x32x32xf32, #ttcore.view<4>, #l1>, memref<4x1x32x32xf32, #ttcore.view<4>, #l1>) -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<4x1x32x32xf32, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>)
     {
      %1 = d2m.get_cb(1) resolution_stage =  compile : <memref<32x32xf32, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      // CHECK: d2m.reserve
      %2 = d2m.reserve %1 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
      // CHECK: scf.for
      // CHECK: scf.for
      // CHECK: scf.if
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

  // Row-major width-concat coalescing factor = 16B / elemBytes; varied via
  // element type across the three funcs below.

  // f32 -> factor 4.
  // CHECK-LABEL: func.func @test_composite_view_row_major_f32_coalesce
  func.func @test_composite_view_row_major_f32_coalesce() -> memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 16, 16>}> : (memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>, memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>) -> memref<4x1x32x32xf32, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<4x1x32x32xf32, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>)
     {
      %1 = d2m.get_cb(1) resolution_stage =  compile : <memref<32x32xf32, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %2 = d2m.reserve %1 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
      // CHECK: [[F32_STEP:%.+]] = arith.constant 4 : index
      // CHECK: scf.for
      // CHECK: scf.for {{.*}} step [[F32_STEP]]
      // CHECK: d2m.dma_read {{.*}}, <4> : (memref<4x1x32x32xf32
      // CHECK: d2m.dma_read {{.*}}, <4> : (memref<4x1x32x32xf32
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<4x1x32x32xf32, #ttcore.view<4>, #l1>, memref<32x32xf32, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      d2m.push %1 : <memref<32x32xf32, #l1>>
    }
    memref.dealloc %alloc_1 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    memref.dealloc %alloc_0 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    return %alloc_2 : memref<4x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
  }

  // bf16 -> factor 8.
  // CHECK-LABEL: func.func @test_composite_view_row_major_bf16_coalesce
  func.func @test_composite_view_row_major_bf16_coalesce() -> memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 16, 16>}> : (memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>, memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>) -> memref<4x1x32x32xbf16, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<4x1x32x32xbf16, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>)
     {
      %1 = d2m.get_cb(1) resolution_stage =  compile : <memref<32x32xbf16, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %2 = d2m.reserve %1 : <memref<32x32xbf16, #l1>> -> memref<32x32xbf16, #l1>
      // CHECK: [[BF16_STEP:%.+]] = arith.constant 8 : index
      // CHECK: scf.for
      // CHECK: scf.for {{.*}} step [[BF16_STEP]]
      // CHECK: d2m.dma_read {{.*}}, <8> : (memref<4x1x32x32xbf16
      // CHECK: d2m.dma_read {{.*}}, <8> : (memref<4x1x32x32xbf16
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<4x1x32x32xbf16, #ttcore.view<4>, #l1>, memref<32x32xbf16, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      d2m.push %1 : <memref<32x32xbf16, #l1>>
    }
    memref.dealloc %alloc_1 : memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    memref.dealloc %alloc_0 : memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
    return %alloc_2 : memref<4x1x32x32xbf16, #ttcore.shard<64x2, 1>, #l1>
  }

  // i8 -> factor 16.
  // CHECK-LABEL: func.func @test_composite_view_row_major_u8_coalesce
  func.func @test_composite_view_row_major_u8_coalesce() -> memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1> {
    %alloc_0 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>
    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%alloc_0, %alloc_1) <{dim = 1 : si32, logicalSizes = array<i64: 16, 16>}> : (memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>, memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>) -> memref<4x1x32x32xi8, #ttcore.view<4>, #l1>
    %alloc_2 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<4x1x32x32xi8, #ttcore.view<4>, #l1>)
        outs(%alloc_2 : memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>)
     {
      %1 = d2m.get_cb(1) resolution_stage =  compile : <memref<32x32xi8, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %2 = d2m.reserve %1 : <memref<32x32xi8, #l1>> -> memref<32x32xi8, #l1>
      // CHECK: [[U8_STEP:%.+]] = arith.constant 16 : index
      // CHECK: scf.for
      // CHECK: scf.for {{.*}} step [[U8_STEP]]
      // CHECK: d2m.dma_read {{.*}}, <16> : (memref<4x1x32x32xi8
      // CHECK: d2m.dma_read {{.*}}, <16> : (memref<4x1x32x32xi8
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<4x1x32x32xi8, #ttcore.view<4>, #l1>, memref<32x32xi8, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      d2m.push %1 : <memref<32x32xi8, #l1>>
    }
    memref.dealloc %alloc_1 : memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>
    memref.dealloc %alloc_0 : memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>
    return %alloc_2 : memref<4x1x32x32xi8, #ttcore.shard<32x1, 1>, #l1>
  }

  // 49 single-tile views of one 7x7 source memref fanout into composite_view
  // checks 48 nested scf.if + 49 dma_read.
  // CHECK-LABEL: func.func @test_composite_view_49_inputs
  func.func @test_composite_view_49_inputs() -> memref<1x49x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
    %src = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>

    %v_0 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_1 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (0, 1, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_2 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (0, 2, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_3 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (0, 3, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_4 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (0, 4, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_5 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (0, 5, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_6 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (0, 6, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_7 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (1, 0, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_8 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (1, 1, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_9 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (1, 2, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_10 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (1, 3, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_11 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (1, 4, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_12 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (1, 5, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_13 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (1, 6, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_14 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (2, 0, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_15 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (2, 1, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_16 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (2, 2, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_17 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (2, 3, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_18 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (2, 4, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_19 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (2, 5, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_20 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (2, 6, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_21 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (3, 0, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_22 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (3, 1, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_23 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (3, 2, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_24 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (3, 3, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_25 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (3, 4, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_26 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (3, 5, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_27 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (3, 6, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_28 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (4, 0, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_29 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (4, 1, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_30 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (4, 2, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_31 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (4, 3, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_32 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (4, 4, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_33 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (4, 5, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_34 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (4, 6, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_35 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (5, 0, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_36 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (5, 1, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_37 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (5, 2, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_38 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (5, 3, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_39 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (5, 4, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_40 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (5, 5, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_41 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (5, 6, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_42 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (6, 0, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_43 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (6, 1, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_44 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (6, 2, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_45 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (6, 3, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_46 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (6, 4, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_47 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (6, 5, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %v_48 = d2m.view_layout %src remapping = affine_map<(d0, d1, d2, d3) -> (6, 6, d2, d3)> : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>

    // CHECK-NOT: d2m.composite_view
    %0 = "d2m.composite_view"(%v_0, %v_1, %v_2, %v_3, %v_4, %v_5, %v_6, %v_7, %v_8, %v_9, %v_10, %v_11, %v_12, %v_13, %v_14, %v_15, %v_16, %v_17, %v_18, %v_19, %v_20, %v_21, %v_22, %v_23, %v_24, %v_25, %v_26, %v_27, %v_28, %v_29, %v_30, %v_31, %v_32, %v_33, %v_34, %v_35, %v_36, %v_37, %v_38, %v_39, %v_40, %v_41, %v_42, %v_43, %v_44, %v_45, %v_46, %v_47, %v_48) <{dim = 1 : si32}> : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>) -> memref<1x49x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    %alloc_out = memref.alloc() {address = 505120 : i64, alignment = 16 : i64} : memref<1x49x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x49>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
        ins(%0 : memref<1x49x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>)
        outs(%alloc_out : memref<1x49x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^datamovement0:
      %1 = d2m.get_cb(1) : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      // CHECK: d2m.reserve
      %2 = d2m.reserve %1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      // CHECK-COUNT-48: scf.if
      // CHECK-NOT: scf.if
      // COUNT-LABEL: func.func @test_composite_view_49_inputs
      // COUNT-COUNT-49: d2m.dma_read
      // COUNT-NOT: d2m.dma_read
      %tx = d2m.dma_read %0[%core0, %core1], %2, <0> : (memref<1x49x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %tx : !d2m.mem_tx<read>
      // CHECK: d2m.push
      d2m.push %1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    }
    memref.dealloc %src : memref<7x7x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    return %alloc_out : memref<1x49x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  }
}
