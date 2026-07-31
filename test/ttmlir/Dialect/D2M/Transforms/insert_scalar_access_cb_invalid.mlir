// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-scalar-access-cb --split-input-file --verify-diagnostics %s

// A single wait/pop pair has to fire at the same cadence as the transfer that
// fills the CB. These shapes cannot be expressed with one pair, and silently
// mismatching the cadence would deadlock, so they are rejected.

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {} {
  // The transfer is inside the loop but the read is outside it: the read is not
  // covered by any pair the loop body could hold.
  func.func @read_outside_filling_block(
      %weights: memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %indices: memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram>) {
    %out = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %wstream = d2m.view_layout %weights remapping = #map4 : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %istream = d2m.view_layout %indices remapping = #map4 : memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram> -> memref<1x1x1x4xi32, #ttcore.view<4>, #dram>
    %idx_cb = memref.alloc() {address = 20480 : i64, alignment = 16 : i64} : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
    %w_cb = memref.alloc() {address = 24576 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // expected-error @+1 {{is scalar-read outside the block that fills it}}
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%istream, %wstream : memref<1x1x1x4xi32, #ttcore.view<4>, #dram>, memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%idx_cb, %w_cb : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %icb = d2m.get_cb(3) : !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>

      scf.for %it = %c0 to %c4 step %c1 {
        d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      }
      %raw = memref.load %idx_cb[%c0, %c0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %row = arith.index_cast %raw : i32 to index
      d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }
}

// -----

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {} {
  // The read precedes the transfer, so a wait placed before it could not observe
  // the transfer's push.
  func.func @read_before_transfer(
      %weights: memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %indices: memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram>) {
    %out = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %wstream = d2m.view_layout %weights remapping = #map4 : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %istream = d2m.view_layout %indices remapping = #map4 : memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram> -> memref<1x1x1x4xi32, #ttcore.view<4>, #dram>
    %idx_cb = memref.alloc() {address = 20480 : i64, alignment = 16 : i64} : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
    %w_cb = memref.alloc() {address = 24576 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // expected-error @+1 {{is scalar-read before the transfer that fills it}}
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%istream, %wstream : memref<1x1x1x4xi32, #ttcore.view<4>, #dram>, memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%idx_cb, %w_cb : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %icb = d2m.get_cb(3) : !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>

      %raw = memref.load %idx_cb[%c0, %c0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %row = arith.index_cast %raw : i32 to index
      d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }
}

// -----

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {} {
  // Nothing on this thread fills the scalar-read CB, so there is no push for a
  // wait to observe and no way to synchronize the read. Reading anyway would race
  // against whatever does fill it.
  func.func @no_transfer_on_read_cb(
      %weights: memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
      %indices: memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram>) {
    %out = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %wstream = d2m.view_layout %weights remapping = #map4 : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %istream = d2m.view_layout %indices remapping = #map4 : memref<1x1x1x4xi32, #ttcore.shard<16x4, 1>, #dram> -> memref<1x1x1x4xi32, #ttcore.view<4>, #dram>
    %idx_cb = memref.alloc() {address = 20480 : i64, alignment = 16 : i64} : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
    %w_cb = memref.alloc() {address = 24576 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>

    // expected-error @+1 {{is scalar-read on a datamovement thread that does not fill it}}
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%istream, %wstream : memref<1x1x1x4xi32, #ttcore.view<4>, #dram>, memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        additionalArgs(%idx_cb, %w_cb : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      %core0 = d2m.core_index(0) : index
      %raw = memref.load %idx_cb[%c0, %core0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %row = arith.index_cast %raw : i32 to index
      d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }
}
