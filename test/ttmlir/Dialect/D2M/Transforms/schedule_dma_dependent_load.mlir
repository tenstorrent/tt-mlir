// RUN: ttmlir-opt --ttcore-register-device --d2m-schedule-dma %s | FileCheck %s

// A dependent load reads an index out of an L1 circular buffer and uses it to
// address a later transfer. The reader takes no part in the CB wait/pop
// protocol, so the only thing that orders it after the transfer filling that CB
// is being on the same data movement thread. d2m-schedule-dma must therefore
// keep the index CB and the CBs of the transfers consuming the loaded value
// together when it load-balances CBs across threads.

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {} {
  // The index CB (port 3) and the gathered-weights CB (port 4) form one
  // scheduling unit; the independent output CB (port 5) is free to go elsewhere.
  //
  // CHECK-LABEL: func.func @dependent_load_pins_producer_to_consumer
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement, dm_core = {{[01]}}>, #d2m.thread<datamovement, dm_core = {{[01]}}>, #d2m.thread<compute>]
  //
  // First DM region: the index CB (3) and the dependent transfer's CB (4)
  // together, with the scalar read between them. Not the independent store.
  // CHECK: d2m.get_cb(3)
  // CHECK: d2m.get_cb(4)
  // CHECK: d2m.remote_load {{.*}} into
  // CHECK: memref.load
  // CHECK: d2m.remote_load {{.*}} into
  // CHECK-NOT: d2m.remote_store
  //
  // Second DM region: only the independent store, and no scalar read.
  // CHECK: }, {
  // CHECK-NOT: memref.load
  // CHECK: d2m.get_cb(5)
  // CHECK: d2m.remote_store
  func.func @dependent_load_pins_producer_to_consumer(
      %weights: memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %indices: memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram>) {
    %out = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %wstream = d2m.view_layout %weights remapping = #map4 : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %istream = d2m.view_layout %indices remapping = #map4 : memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram> -> memref<1x1x1x4xi32, #ttcore.view<4>, #dram>
    %idx_cb = memref.alloc() {address = 20480 : i64, alignment = 16 : i64} : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
    %w_cb = memref.alloc() {address = 24576 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %o_cb = memref.alloc() {address = 90112 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%istream, %wstream : memref<1x1x1x4xi32, #ttcore.view<4>, #dram>, memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%idx_cb, %w_cb, %o_cb : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %icb = d2m.get_cb(3) : !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %ocb = d2m.get_cb(5) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index

      d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %raw = memref.load %idx_cb[%c0, %core0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %row = arith.index_cast %raw : i32 to index
      d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      d2m.remote_store %out[%core0, %core1] from %ocb : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }

  // The loaded value can reach the dependent transfer through a loop-carried
  // accumulator, i.e. via scf.yield rather than a plain op result. The affinity
  // walk has to follow that, or the two CBs look independent and get split
  // across threads.
  //
  // CHECK-LABEL: func.func @dependent_load_through_loop_carried_value
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement, dm_core = {{[01]}}>, #d2m.thread<datamovement, dm_core = {{[01]}}>, #d2m.thread<compute>]
  // First DM region: the index transfer, the loop that reads it, and the
  // transfer that depends on the accumulated value -- all together.
  // CHECK: d2m.remote_load %{{.*}} into
  // CHECK: scf.for
  // CHECK: memref.load
  // CHECK: d2m.remote_load %{{.*}} into
  // CHECK-NOT: d2m.remote_store
  // CHECK: }, {
  func.func @dependent_load_through_loop_carried_value(
      %weights: memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %indices: memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram>) {
    %out = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %wstream = d2m.view_layout %weights remapping = #map4 : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %istream = d2m.view_layout %indices remapping = #map4 : memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram> -> memref<1x1x1x4xi32, #ttcore.view<4>, #dram>
    %idx_cb = memref.alloc() {address = 20480 : i64, alignment = 16 : i64} : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
    %w_cb = memref.alloc() {address = 24576 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %o_cb = memref.alloc() {address = 90112 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%istream, %wstream : memref<1x1x1x4xi32, #ttcore.view<4>, #dram>, memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%idx_cb, %w_cb, %o_cb : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %zero = arith.constant 0 : i32
      %icb = d2m.get_cb(3) : !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %ocb = d2m.get_cb(5) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index

      d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %acc = scf.for %k = %c0 to %c4 step %c1 iter_args(%a = %zero) -> (i32) {
        %v = memref.load %idx_cb[%c0, %k] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
        %s = arith.addi %a, %v : i32
        scf.yield %s : i32
      }
      %row = arith.index_cast %acc : i32 to index
      d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      d2m.remote_store %out[%core0, %core1] from %ocb : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }

  // A scalar access on a buffer that is not a generic operand cannot be
  // attributed to a CB port, so it cannot be filtered per thread. Splitting
  // would replicate it into every thread, so the pass declines to split at all
  // and leaves a single DM thread.
  //
  // CHECK-LABEL: func.func @unattributable_scalar_access_blocks_split
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement, dm_core = {{[01]}}>, #d2m.thread<compute>]
  func.func @unattributable_scalar_access_blocks_split(
      %weights: memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %indices: memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram>) {
    %out = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %wstream = d2m.view_layout %weights remapping = #map4 : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %istream = d2m.view_layout %indices remapping = #map4 : memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram> -> memref<1x1x1x4xi32, #ttcore.view<4>, #dram>
    %idx_cb = memref.alloc() {address = 20480 : i64, alignment = 16 : i64} : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
    %w_cb = memref.alloc() {address = 24576 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
    %o_cb = memref.alloc() {address = 90112 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%istream, %wstream : memref<1x1x1x4xi32, #ttcore.view<4>, #dram>, memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%idx_cb, %w_cb, %o_cb : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %icb = d2m.get_cb(3) : !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %ocb = d2m.get_cb(5) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      // A region-local scratch buffer: not a generic operand, so not a CB port.
      %scratch = memref.alloc() {d2m.scratch_buffer} : memref<4xi32, #l1>

      d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %raw = memref.load %scratch[%c0] : memref<4xi32, #l1>
      %row = arith.index_cast %raw : i32 to index
      d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      d2m.remote_store %out[%core0, %core1] from %ocb : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }
}
