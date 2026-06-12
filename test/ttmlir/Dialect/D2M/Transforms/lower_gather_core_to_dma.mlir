// RUN: ttmlir-opt --ttcore-register-device \
// RUN:            --d2m-preallocate-mcast-semaphores \
// RUN:            --d2m-lower-load-store-ops-to-dma %s | FileCheck %s

// d2m.gather_core lowers to:
//
//   if isInGroup:
//     if isCollector:
//       semaphore_wait %sourceReady, groupVolume - 1
//       for sy in groupY: for sx in groupX:
//         %tx = dma_read %src[] core[sy, sx], %dst, <0>
//         dma_wait %tx
//       ; collectorDone fan-out: mcast set (general) or loop of unicast
//       ; incs (1x1 / 1xN / Nx1) -- see LowerLoadStoreOpsToDMA.cpp.
//     else:
//       semaphore_inc  %sourceReady,  1, core[collector]
//       semaphore_wait %collectorDone, 1
//
// In CB form the src CB wait/pop straddles the isInGroup gate (every
// group core consumes one src-CB element).

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {
  // 1x4 group at (0,0), collector (0,0). numSources = 3.
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

      // Outer scf.if = isInGroup, inner = isCollector.
      // CHECK: d2m.core_index(0)
      // CHECK: d2m.core_index(1)
      // CHECK: scf.if %{{.*}} {
      // CHECK-NEXT: scf.if %{{.*}} {
      //   Collector (numSources = 3 = 4 - 1).
      //   CHECK:     d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      //   CHECK:     scf.for
      //   CHECK:       scf.for %[[SX:.*]] =
      //   CHECK:         %[[TX:.*]] = d2m.dma_read %[[SRC]][] core[%{{.*}}, %[[SX]]], %[[DST]], <0>
      //   CHECK:         d2m.dma_wait %[[TX]]
      //   1xN: loop of unicast incs (no mcast set).
      //   CHECK:     scf.for %[[INCX:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
      //   CHECK:       d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %[[INCX]]]
      //   CHECK-NOT: d2m.semaphore_set
      // CHECK:   } else {
      //   Source.
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

  // CB-form: src CB wait/pop on every group core; dst CB reserve/push
  // only on the collector. Reads + semaphore handshake unchanged from
  // implicit form.
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

      // src CB wait outside the gates; dst CB reserve/push inside the
      // collector branch only.
      // CHECK: %[[SRC_LOCAL:.*]] = d2m.wait %[[SRC_CB]]
      // CHECK: scf.if %{{.*}} {
      // CHECK-NEXT: scf.if %{{.*}} {
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
      //   CHECK:     d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}]
      //   CHECK:     d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      //   CHECK-NOT:     d2m.reserve
      //   CHECK-NOT:     d2m.push
      // CHECK:   }
      // CHECK: }
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

  // 2x2 group: collectorDone fan-out keeps the single mcast semaphore_set.
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

  // 1x1 group: single unicast inc, no scf.for for the fan-out.
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
      //   Single unicast inc; any scf.for here is from the DMA loop only.
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

  // 4x1 group: loop over Y, one unicast inc per group core.
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
