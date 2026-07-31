// RUN: ttmlir-opt --ttcore-register-device --d2m-optimize-dma %s | FileCheck %s

// d2m-optimize-dma coalesces transfers by sinking each dma_wait (with its
// push/pop) as late as legal. A scalar L1 read takes no part in the CB
// handshake, so the barrier is the only thing ordering it against the transfer
// that fills the buffer: sinking past it would let a dependent load observe data
// that has not landed.

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {} {
  // The index transfer's barrier must stay above the memref.load; the dependent
  // transfer's barrier is still free to sink to the end.
  //
  // CHECK-LABEL: func.func @barrier_does_not_sink_past_dependent_load
  func.func @barrier_does_not_sink_past_dependent_load(
      %weights: memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %indices: memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram>) {
    %out = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %wstream = d2m.view_layout %weights remapping = #map4 : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %istream = d2m.view_layout %indices remapping = #map4 : memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram> -> memref<1x1x1x4xi32, #ttcore.view<4>, #dram>
    %idx_cb = memref.alloc() {address = 20480 : i64, alignment = 16 : i64} : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
    %w_cb = memref.alloc() {address = 24576 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%istream, %wstream : memref<1x1x1x4xi32, #ttcore.view<4>, #dram>, memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%idx_cb, %w_cb : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %icb = d2m.get_cb(3) : !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %core0 = d2m.core_index(0) : index

      // CHECK: [[ILOCAL:%.+]] = d2m.reserve
      // CHECK: [[ITX:%.+]] = d2m.dma_read
      // The index barrier stays put -- it may not sink past the scalar read.
      // CHECK-NEXT: d2m.dma_wait [[ITX]] : !d2m.mem_tx<read>
      // CHECK: memref.load
      // CHECK: [[WTX:%.+]] = d2m.dma_read
      // CHECK: d2m.dma_wait [[WTX]] : !d2m.mem_tx<read>
      %ilocal = d2m.reserve %icb : <memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>> -> memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %itx = d2m.dma_read %istream[%c0, %c0], %ilocal, <0> : (memref<1x1x1x4xi32, #ttcore.view<4>, #dram>, memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %itx : !d2m.mem_tx<read>
      d2m.push %icb : <memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>

      %raw = memref.load %idx_cb[%c0, %core0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %row = arith.index_cast %raw : i32 to index

      %wlocal = d2m.reserve %wcb : <memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      %wtx = d2m.dma_read %wstream[%row, %c0], %wlocal, <0> : (memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %wtx : !d2m.mem_tx<read>
      d2m.push %wcb : <memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }

  // A scalar read of an *unrelated* CB is no reason to hold a barrier back, so
  // the usual coalescing still happens: both dma_reads issue, then both waits.
  //
  // CHECK-LABEL: func.func @barrier_sinks_past_unrelated_scalar_load
  func.func @barrier_sinks_past_unrelated_scalar_load(
      %a: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %b: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %out = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %astream = d2m.view_layout %a remapping = #map4 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %bstream = d2m.view_layout %b remapping = #map4 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %a_cb = memref.alloc() {address = 20480 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %b_cb = memref.alloc() {address = 90112 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %scalar_cb = memref.alloc() {address = 159744 : i64, alignment = 16 : i64} : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%astream, %bstream : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%a_cb, %b_cb, %scalar_cb : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %acb = d2m.get_cb(3) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %bcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %core0 = d2m.core_index(0) : index

      // CHECK: [[ATX:%.+]] = d2m.dma_read
      // CHECK: memref.load
      // CHECK: [[BTX:%.+]] = d2m.dma_read
      // CHECK-NEXT: d2m.dma_wait [[ATX]] : !d2m.mem_tx<read>
      // CHECK-NEXT: d2m.push
      // CHECK-NEXT: d2m.dma_wait [[BTX]] : !d2m.mem_tx<read>
      %alocal = d2m.reserve %acb : <memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      %atx = d2m.dma_read %astream[%c0, %c0], %alocal, <0> : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %atx : !d2m.mem_tx<read>
      d2m.push %acb : <memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>

      // Reads CB port 5, which no transfer here touches.
      %raw = memref.load %scalar_cb[%c0, %core0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>

      %blocal = d2m.reserve %bcb : <memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
      %btx = d2m.dma_read %bstream[%c0, %c0], %blocal, <0> : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) -> !d2m.mem_tx<read>
      d2m.dma_wait %btx : !d2m.mem_tx<read>
      d2m.push %bcb : <memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }
}
