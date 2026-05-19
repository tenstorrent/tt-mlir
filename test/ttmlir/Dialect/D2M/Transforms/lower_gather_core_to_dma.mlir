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
}
