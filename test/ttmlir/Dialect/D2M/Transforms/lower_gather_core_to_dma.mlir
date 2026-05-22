// RUN: ttmlir-opt --ttcore-register-device \
// RUN:            --d2m-preallocate-mcast-semaphores \
// RUN:            --d2m-lower-load-store-ops-to-dma %s | FileCheck %s

// d2m.gather_core lowers (inside its hosting d2m.generic data-movement
// region) to:
//
//   if isInGroup:                                                   // gate
//     if isCollector:
//       semaphore_wait %sourceReady,  numSources                    // sources signal in
//       for sy in groupY:
//         for sx in groupX:
//           %tx = dma_read %src[] core[sy, sx], %dst, <0>           // shard-level read
//           dma_wait %tx
//       ; collectorDone fan-out:
//       ;   groupShape statically 1x1 -> one unicast inc to the group core
//       ;   groupShape statically 1xN or Nx1 -> loop of unicast incs
//       ;   else -> single mcast semaphore_set
//     else:
//       semaphore_inc  %sourceReady,  1, core[collector]            // signal collector
//       semaphore_wait %collectorDone, 1                            // wait for completion
//
// numSources = groupVolume - 1 because the collector does not signal itself
// and pulls its own data via the self-iteration of the loop.
//
// The outer `if isInGroup` gate (V2) excludes cores outside the gather
// group from the protocol entirely. Without it, those cores would still
// execute the source-side path, over-saturate sourceReady and deadlock on
// collectorDone (which is only fan-out to in-group cores). The src CB
// `d2m.wait`/`d2m.pop` handshake (when in explicit CB form) is emitted
// unconditionally outside the if, because the src CB allocation is per-
// core-uniform today and every core's upstream compute thread pushes one
// element into it.
//
// The degenerate-axis (1xN / Nx1) path exists because the downstream
// emitc.expression wrapping of ttkernel::experimental::get_noc_multicast_addr
// cannot roundtrip duplicate start/end SSA values on the degenerate axis
// (Canonicalize folds the `start + 1 - 1` end-coord arithmetic, CSE then
// collapses the two convert-logical calls, and emitc.expression's printer
// shadows region args with operand names, producing two block args with
// the same name -> AsmParser rejects the IR). The loop-of-incs form is
// observable-equivalent inside this protocol because collectorDone is
// zero-initialized, reset on the source's wait, and only ever published
// with value 1.

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {
  // Group: 1x4 starting at (0,0), collector at (0,0). groupVolume=4, so
  // numSources = 3.
  // CHECK-LABEL: func.func @test_lower_gather_core
  // CHECK-NOT: d2m.gather_core
  func.func @test_lower_gather_core(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>,
                 indexing_maps = [#map, #map],
                 iterator_types = [#parallel, #parallel],
                 threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)  {
    ^datamovement0:
      // CHECK: %[[SRC:.*]] = memref.alloc()
      // CHECK: %[[DST:.*]] = memref.alloc()
      %src = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %dst = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // isCollector = (core_index(0) == 0) && (core_index(1) == 0)
      // isInGroup   = (core_index(0) ∈ [0, 1)) && (core_index(1) ∈ [0, 4))
      // CHECK: d2m.core_index(0)
      // CHECK: d2m.core_index(1)
      // The outer scf.if is the isInGroup gate; the inner is isCollector.
      // CHECK: scf.if %{{.*}} {
      // CHECK-NEXT: scf.if %{{.*}} {
      //   Collector branch.
      //   numSources = groupVolume - 1 = 4 - 1 = 3.
      //   CHECK:     d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      //   CHECK:     scf.for
      //   CHECK:       scf.for %[[SX:.*]] =
      //   CHECK:         %[[TX:.*]] = d2m.dma_read %[[SRC]][] core[%{{.*}}, %[[SX]]], %[[DST]], <0>
      //   CHECK:         d2m.dma_wait %[[TX]]
      //   1x4 group -> single scf.for over X, one unicast inc per group core.
      //   CHECK:     scf.for %[[INCX:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
      //   CHECK:       d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %[[INCX]]]
      //   CHECK-NOT: d2m.semaphore_set
      // CHECK:   } else {
      //   Source branch.
      //   CHECK:     d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}]
      //   CHECK:     d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      // CHECK:   }
      // CHECK: }
      d2m.gather_core %src into %dst
        group [%c0, %c0] shape [%c1, %c4] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }, {
    ^compute0:
    }
    return
  }

  // CB-form gather_core lowering: src is consumed via d2m.wait/d2m.pop on
  // every core; the collector additionally d2m.reserves and d2m.pushes the
  // dst CB (option (c): allocate the dst CB everywhere, only the collector
  // executes its CB ops). The DMA reads and the semaphore handshake are the
  // same as in the implicit-form lowering above.
  //
  // CHECK-LABEL: func.func @test_lower_gather_core_cb_form
  // CHECK-NOT: d2m.gather_core
  func.func @test_lower_gather_core_cb_form(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    %srcAlloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1_>
    %dstAlloc = memref.alloc() {address = 6144 : i64, alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1_>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>,
                 indexing_maps = [#map, #map],
                 iterator_types = [#parallel, #parallel],
                 threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        additionalArgs(%srcAlloc, %dstAlloc : memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1_>, memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1_>) {
    ^datamovement0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      // CHECK: %[[SRC_CB:.*]] = d2m.get_cb(2)
      // CHECK: %[[DST_CB:.*]] = d2m.get_cb(3)
      %srcCb = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1_>>
      %dstCb = d2m.get_cb(3) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1_>>

      // CB-form lowering shape (wait/pop on src CB stay outside the
      // isInGroup gate; reserve/push on dst CB live in the collector
      // branch as before, so two levels of scf.if are visible inside the
      // generic).
      // CHECK: %[[SRC_LOCAL:.*]] = d2m.wait %[[SRC_CB]]
      // CHECK: scf.if %{{.*}} {
      // CHECK-NEXT: scf.if %{{.*}} {
      //   Collector branch: reserve dst CB, wait for sources, DMA-read,
      //   loop of unicast incs (1x4 degenerate), push dst CB.
      //   CHECK:     %[[DST_LOCAL:.*]] = d2m.reserve %[[DST_CB]]
      //   CHECK:     d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      //   CHECK:     scf.for
      //   CHECK:       scf.for
      //   CHECK:         %[[TX:.*]] = d2m.dma_read %[[SRC_LOCAL]][] core[%{{.*}}, %{{.*}}], %[[DST_LOCAL]], <0>
      //   CHECK:         d2m.dma_wait %[[TX]]
      //   CHECK:     scf.for %[[INCX:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
      //   CHECK:       d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %[[INCX]]]
      //   CHECK-NOT: d2m.semaphore_set
      //   CHECK:     d2m.push %[[DST_CB]]
      // CHECK:   } else {
      //   Source branch: notify collector, wait for the per-core inc. No dst CB ops here.
      //   CHECK:     d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}]
      //   CHECK:     d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      //   CHECK-NOT:     d2m.reserve
      //   CHECK-NOT:     d2m.push
      // CHECK:   }
      // CHECK: }
      // After the branch, every core pops the src CB.
      // CHECK: d2m.pop %[[SRC_CB]]
      d2m.gather_core from %srcCb into %dstCb
        group [%c0, %c0] shape [%c1, %c4] collector [%c0, %c0]
        : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1_>>,
          !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1_>>
    }, {
    ^compute0:
    }
    return
  }

  // Non-degenerate (2x2) gather group: the collectorDone fan-out keeps the
  // single mcast semaphore_set form, no per-core loop of incs.
  // CHECK-LABEL: func.func @test_lower_gather_core_2x2
  // CHECK-NOT: d2m.gather_core
  func.func @test_lower_gather_core_2x2(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>,
                 indexing_maps = [#map, #map],
                 iterator_types = [#parallel, #parallel],
                 threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)  {
    ^datamovement0:
      %src = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %dst = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      // CHECK: scf.if %{{.*}} {
      // CHECK-NEXT: scf.if %{{.*}} {
      //   Collector branch: 2x2 mcast fan-out stays as a single semaphore_set.
      //   CHECK:     d2m.semaphore_set %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}] mcast[%{{.*}}, %{{.*}}]
      //   CHECK-NOT: d2m.semaphore_inc %{{[^,]*}}, %{{[^,]*}}, core[
      // CHECK:   } else {
      d2m.gather_core %src into %dst
        group [%c0, %c0] shape [%c2, %c2] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }, {
    ^compute0:
    }
    return
  }

  // 1x1 gather group (degenerate on both axes): a single unicast inc, no scf.for.
  // CHECK-LABEL: func.func @test_lower_gather_core_1x1
  // CHECK-NOT: d2m.gather_core
  func.func @test_lower_gather_core_1x1(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>,
                 indexing_maps = [#map, #map],
                 iterator_types = [#parallel, #parallel],
                 threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)  {
    ^datamovement0:
      %src = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %dst = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // CHECK: scf.if %{{.*}} {
      // CHECK-NEXT: scf.if %{{.*}} {
      //   1x1 group: emit a single unicast inc to the (only) group core.
      //   Any scf.for ops here come from the DMA loop only.
      //   CHECK:     d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}]
      //   CHECK-NOT: d2m.semaphore_set
      // CHECK:   } else {
      d2m.gather_core %src into %dst
        group [%c0, %c0] shape [%c1, %c1] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }, {
    ^compute0:
    }
    return
  }

  // Nx1 gather group: single scf.for over Y, one unicast inc per group core.
  // CHECK-LABEL: func.func @test_lower_gather_core_4x1
  // CHECK-NOT: d2m.gather_core
  func.func @test_lower_gather_core_4x1(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>,
                 indexing_maps = [#map, #map],
                 iterator_types = [#parallel, #parallel],
                 threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)  {
    ^datamovement0:
      %src = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %dst = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      // CHECK: scf.if %{{.*}} {
      // CHECK-NEXT: scf.if %{{.*}} {
      //   Nx1 group: walk the non-degenerate axis (Y) and emit one inc per core.
      //   CHECK:     scf.for %[[INCY:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
      //   CHECK:       d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%[[INCY]], %{{.*}}]
      //   CHECK-NOT: d2m.semaphore_set
      // CHECK:   } else {
      d2m.gather_core %src into %dst
        group [%c0, %c0] shape [%c4, %c1] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }, {
    ^compute0:
    }
    return
  }
}
