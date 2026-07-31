// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-scalar-access-cb %s | FileCheck %s

// A scalar L1 read is a CB consumer, but it is not discovered by
// d2m-insert-compute-cb. Without a wait/pop pair the transfer half reserves and
// pushes with nothing popping, so the read addresses the front of the buffer
// instead of the page just written, and a loop exhausts the CB and blocks in
// reserve. This pass supplies the missing pair, at the transfer's loop depth.

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module attributes {} {
  // Straight line: bracket the read, and rewire it onto the wait result so the
  // acquisition is an SSA dependency rather than just program order.
  //
  // CHECK-LABEL: func.func @straight_line
  func.func @straight_line(
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

      // CHECK: d2m.remote_load {{.*}} into %[[ICB:[0-9a-z_]+]]
      // CHECK-NEXT: %[[BUF:[0-9a-z_]+]] = d2m.wait %[[ICB]]
      // CHECK-NEXT: memref.load %[[BUF]]
      // CHECK-NEXT: d2m.pop %[[ICB]]
      d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %raw = memref.load %idx_cb[%c0, %core0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %row = arith.index_cast %raw : i32 to index
      d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }

  // Several reads of one CB collapse to a single wait/pop pair, bracketing the
  // first and last of them.
  //
  // CHECK-LABEL: func.func @several_reads_one_pair
  func.func @several_reads_one_pair(
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

      // The pop lands right after the last read, not after the arithmetic: the
      // loaded values are already in registers, so the page can be released.
      // CHECK: d2m.remote_load {{.*}} into %[[ICBM:[0-9a-z_]+]]
      // CHECK-NEXT: %[[BUFM:[0-9a-z_]+]] = d2m.wait %[[ICBM]]
      // CHECK-NEXT: memref.load %[[BUFM]]
      // CHECK-NEXT: memref.load %[[BUFM]]
      // CHECK-NEXT: d2m.pop %[[ICBM]]
      // CHECK-NEXT: arith.addi
      d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %raw = memref.load %idx_cb[%c0, %core0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %raw2 = memref.load %idx_cb[%c0, %c0] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
      %sum = arith.addi %raw, %raw2 : i32
      %row = arith.index_cast %sum : i32 to index
      d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
    }, {
    ^compute0:
    }
    return
  }

  // The index block is re-filled every iteration, so the pair goes inside the
  // loop and is balanced per iteration.
  //
  // CHECK-LABEL: func.func @refilled_each_iteration
  func.func @refilled_each_iteration(
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
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %icb = d2m.get_cb(3) : !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>

      // CHECK: scf.for
      // CHECK: d2m.remote_load {{.*}} into %[[ICB2:[0-9a-z_]+]]
      // CHECK-NEXT: %[[BUF2:[0-9a-z_]+]] = d2m.wait %[[ICB2]]
      // CHECK-NEXT: memref.load %[[BUF2]]
      // CHECK-NEXT: d2m.pop %[[ICB2]]
      scf.for %it = %c0 to %c4 step %c1 {
        d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
        %raw = memref.load %idx_cb[%c0, %it] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
        %row = arith.index_cast %raw : i32 to index
        d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      }
    }, {
    ^compute0:
    }
    return
  }

  // The index block is filled once and indexed many times. The pair must match
  // the *transfer's* depth -- a pair inside the loop would wait a second time on
  // a CB that was pushed once -- so it brackets the whole loop.
  //
  // CHECK-LABEL: func.func @filled_once_read_in_loop
  func.func @filled_once_read_in_loop(
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
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %icb = d2m.get_cb(3) : !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      %wcb = d2m.get_cb(4) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>

      // CHECK: d2m.remote_load {{.*}} into %[[ICB3:[0-9a-z_]+]]
      // CHECK-NEXT: %[[BUF3:[0-9a-z_]+]] = d2m.wait %[[ICB3]]
      // CHECK-NEXT: scf.for
      // CHECK: memref.load %[[BUF3]]
      // CHECK: }
      // CHECK-NEXT: d2m.pop %[[ICB3]]
      d2m.remote_load %istream[%c0, %c0] into %icb : memref<1x1x1x4xi32, #ttcore.view<4>, #dram> into !d2m.cb<memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>>
      scf.for %it = %c0 to %c4 step %c1 {
        %raw = memref.load %idx_cb[%c0, %it] : memref<1x4xi32, #ttcore.cb_layout<16x4, 2>, #l1>
        %row = arith.index_cast %raw : i32 to index
        d2m.remote_load %wstream[%row, %c0] into %wcb : memref<4x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>>
      }
    }, {
    ^compute0:
    }
    return
  }

}
