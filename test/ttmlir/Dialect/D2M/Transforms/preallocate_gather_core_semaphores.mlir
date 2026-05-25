// RUN: ttmlir-opt --ttcore-register-device --d2m-preallocate-mcast-semaphores -o %t %s
// RUN: FileCheck %s --input-file=%t

// d2m-preallocate-mcast-semaphores allocates a fresh local-semaphore pair
// per gather_core, surfaced in the generic's additionalArgs and recorded
// on the op as preallocated_semaphores = [sem0Idx, sem1Idx].

#l1_ = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  // Single gather_core: one preallocated semaphore pair.
  // CHECK-LABEL: func.func @test_single_gather_core
  func.func @test_single_gather_core(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    // ins+outs=2, so the two semaphores land at indices 2 and 3.
    // CHECK: %[[S0:.*]] = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    // CHECK: %[[S1:.*]] = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    // CHECK: d2m.generic
    // CHECK: additionalArgs(%[[S0]], %[[S1]] : !d2m.local_semaphore, !d2m.local_semaphore)
    // CHECK: d2m.gather_core {{.*}} {preallocated_semaphores = [2, 3]}
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
      d2m.gather_core %src into %dst
        group [%c0, %c0] shape [%c1, %c4] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }, {
    ^compute0:
    }
    return
  }

  // Two gather_cores in the same generic: each gets its own pair.
  // CHECK-LABEL: func.func @test_two_gather_cores
  func.func @test_two_gather_cores(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>,
      %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>) {
    // CHECK: %[[S0:.*]] = d2m.create_local_semaphore
    // CHECK: %[[S1:.*]] = d2m.create_local_semaphore
    // CHECK: %[[S2:.*]] = d2m.create_local_semaphore
    // CHECK: %[[S3:.*]] = d2m.create_local_semaphore
    // CHECK: d2m.generic
    // CHECK: additionalArgs(%[[S0]], %[[S1]], %[[S2]], %[[S3]] : !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore)
    // CHECK: d2m.gather_core {{.*}} {preallocated_semaphores = [2, 3]}
    // CHECK: d2m.gather_core {{.*}} {preallocated_semaphores = [4, 5]}
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>,
                 indexing_maps = [#map, #map],
                 iterator_types = [#parallel, #parallel],
                 threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)
        outs(%arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)  {
    ^datamovement0:
      %srcA = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %dstA = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %srcB = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %dstB = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      d2m.gather_core %srcA into %dstA
        group [%c0, %c0] shape [%c1, %c4] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      d2m.gather_core %srcB into %dstB
        group [%c0, %c0] shape [%c1, %c4] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }, {
    ^compute0:
    }
    return
  }

  // gather_core + multicast remote_load in the same generic: each gets its
  // own pair (both kinds are collected uniformly by this pass).
  // CHECK-LABEL: func.func @test_mixed_gather_and_mcast_remote_load
  func.func @test_mixed_gather_and_mcast_remote_load(
      %arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>

    // CHECK: %[[S0:.*]] = d2m.create_local_semaphore
    // CHECK: %[[S1:.*]] = d2m.create_local_semaphore
    // CHECK: %[[S2:.*]] = d2m.create_local_semaphore
    // CHECK: %[[S3:.*]] = d2m.create_local_semaphore
    // CHECK: d2m.generic
    // CHECK: additionalArgs(%[[S0]], %[[S1]], %[[S2]], %[[S3]] : !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore)
    // CHECK: d2m.remote_load {{.*}} {preallocated_semaphores = [2, 3]}
    // CHECK: d2m.gather_core {{.*}} {preallocated_semaphores = [4, 5]}
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>,
                 indexing_maps = [#map, #map],
                 iterator_types = [#parallel, #reduction],
                 threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1_>)  {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
      %scratch_src = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %scratch_dst = memref.alloc() {alignment = 16 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      d2m.remote_load %arg0[%c0, %c0] into %cb0 mcore[%c0, %c0] mshape[%c1, %c4]
        : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
          into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>
      d2m.gather_core %scratch_src into %scratch_dst
        group [%c0, %c0] shape [%c1, %c4] collector [%c0, %c0]
        : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
          memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    }, {
    ^compute0:
    }
    return
  }
}
