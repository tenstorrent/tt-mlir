// RUN: ttmlir-opt --ttcore-register-device --d2m-preallocate-mcast-semaphores --d2m-lower-load-store-ops-to-dma %s | FileCheck %s

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  // Test that multicast remote_load lowering works when mcastShape values are
  // dynamic (not arith.constant). Prior to this fix, the pass extracted constant
  // integer values from arith.constant ops to compute mcast volume statically.
  // Non-constant values were silently treated as dim=1, producing incorrect
  // lowerings. The fix computes mcast volume dynamically via arith.muli.
  //
  // CHECK-LABEL: func.func @test_mcast_dynamic_shape
  func.func @test_mcast_dynamic_shape(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, %mcast_y: index, %mcast_x: index) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
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
          // Both mcastShape dims are dynamic (function arguments, not constants).
          d2m.remote_load %arg0[%0, %1] into %cb0 mcore[%c0, %c0] mshape[%mcast_y, %mcast_x] : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
}

// The remote_load must be replaced by the full mcast lowering.
// CHECK-NOT: d2m.remote_load

// Dynamic mcast volume: arith.muli computes volume from the dynamic mshape dims.
// CHECK: %[[VOL:.*]] = arith.muli
// Dynamic receiver count: arith.subi computes (volume - 1).
// CHECK: %[[NUM_RECV:.*]] = arith.subi %[[VOL]],

// Sender/receiver branch structure.
// CHECK: scf.if
// Sender path: DMA read, wait for receivers ready, multicast write, signal done.
// CHECK: d2m.dma_read
// CHECK: d2m.semaphore_wait %{{.*}}, %[[NUM_RECV]]
// CHECK: d2m.dma_write
// CHECK: d2m.semaphore_set
// Receiver path: signal ready, wait for sender.
// CHECK: d2m.semaphore_inc
// CHECK: d2m.semaphore_wait
