// RUN: ttmlir-opt --ttcore-register-device \
// RUN:            --d2m-preallocate-mcast-semaphores \
// RUN:            --d2m-lower-load-store-ops-to-dma %s | FileCheck %s

// d2m.gather_core lowers (inside its hosting d2m.generic data-movement
// region) to a collector/source handshake:
//
//   if isCollector:
//     semaphore_wait %sourceReady,  numSources                      // sources signal in
//     for sy in groupY:
//       for sx in groupX:
//         %tx = dma_read %src[] core[sy, sx], %dst, <0>             // shard-level read
//         dma_wait %tx
//     semaphore_set  %collectorDone, 1, core[groupStart] mcast[groupShape]
//   else:
//     semaphore_inc  %sourceReady,  1, core[collector]              // signal collector
//     semaphore_wait %collectorDone, 1                              // wait for completion
//
// numSources = groupVolume - 1 because the collector does not signal itself
// and pulls its own data via the self-iteration of the loop.

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
      // CHECK: %[[Y:.*]] = d2m.core_index(0)
      // CHECK: %[[YEQ:.*]] = arith.cmpi eq, %[[Y]],
      // CHECK: %[[X:.*]] = d2m.core_index(1)
      // CHECK: %[[XEQ:.*]] = arith.cmpi eq, %[[X]],
      // CHECK: %[[IS_COLLECTOR:.*]] = arith.andi %[[YEQ]], %[[XEQ]]
      // CHECK: scf.if %[[IS_COLLECTOR]] {
      //   Collector branch.
      //   numSources = groupVolume - 1 = 4 - 1 = 3.
      //   CHECK:   d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      //   CHECK:   scf.for
      //   CHECK:     scf.for %[[SX:.*]] =
      //   CHECK:       %[[TX:.*]] = d2m.dma_read %[[SRC]][] core[%{{.*}}, %[[SX]]], %[[DST]], <0>
      //   CHECK:       d2m.dma_wait %[[TX]]
      //   CHECK:   d2m.semaphore_set %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}] mcast[%{{.*}}, %{{.*}}]
      // CHECK: } else {
      //   Source branch.
      //   CHECK:   d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}]
      //   CHECK:   d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
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

      // CB-form lowering shape:
      // CHECK: %[[SRC_LOCAL:.*]] = d2m.wait %[[SRC_CB]]
      // CHECK: scf.if %{{.*}} {
      //   Collector branch: reserve dst CB, wait for sources, DMA-read, mcast set, push dst CB.
      //   CHECK:   %[[DST_LOCAL:.*]] = d2m.reserve %[[DST_CB]]
      //   CHECK:   d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      //   CHECK:   scf.for
      //   CHECK:     scf.for
      //   CHECK:       %[[TX:.*]] = d2m.dma_read %[[SRC_LOCAL]][] core[%{{.*}}, %{{.*}}], %[[DST_LOCAL]], <0>
      //   CHECK:       d2m.dma_wait %[[TX]]
      //   CHECK:   d2m.semaphore_set %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}] mcast[%{{.*}}, %{{.*}}]
      //   CHECK:   d2m.push %[[DST_CB]]
      // CHECK: } else {
      //   Source branch: notify collector, wait for mcast set. No dst CB ops here.
      //   CHECK:   d2m.semaphore_inc %{{.*}}, %{{.*}}, core[%{{.*}}, %{{.*}}]
      //   CHECK:   d2m.semaphore_wait %{{.*}}, %{{.*}} reset %{{.*}}
      //   CHECK-NOT:   d2m.reserve
      //   CHECK-NOT:   d2m.push
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
}
