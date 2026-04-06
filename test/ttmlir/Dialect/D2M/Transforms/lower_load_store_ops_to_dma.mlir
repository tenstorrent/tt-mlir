// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-load-store-ops-to-dma %s | FileCheck %s

#dram = #ttcore.memory_space<dram>
#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
module attributes {} {
  // CHECK-LABEL: func.func @test_remote_load_explicit_cb
  // CHECK-NOT: d2m.remote_load
  // CHECK-NOT: d2m.remote_store
  func.func @test_remote_load_explicit_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
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
          // CHECK: %[[MEMREF:.*]] = d2m.reserve %{{.*}}
          // CHECK: %[[TX:.*]] = d2m.dma_read %{{.*}}[%{{.*}}, %{{.*}}], %[[MEMREF]], <0>
          // CHECK-NEXT: d2m.dma_wait %[[TX]]
          // CHECK-NEXT: d2m.push %{{.*}}
          d2m.remote_load %stream[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
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
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%0 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %1 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %1 : !ttcore.tile<32x32, f32>
          }
          d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
  // CHECK-LABEL: func.func @test_remote_store_explicit_cb
  // CHECK-NOT: d2m.remote_load
  // CHECK-NOT: d2m.remote_store
  func.func @test_remote_store_explicit_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          // CHECK: %[[MEMREF:.*]] = d2m.wait %{{.*}}
          // CHECK: %[[TX:.*]] = d2m.dma_write %[[MEMREF]], %{{.*}}[%{{.*}}, %{{.*}}], <0>
          // CHECK-NEXT: d2m.dma_wait %[[TX]]
          // CHECK-NEXT: d2m.pop %{{.*}}
          d2m.remote_store %stream[%0, %1] from %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          affine.for %arg4 = 0 to 2 {
            affine.for %arg5 = 0 to 4 {
              %1 = affine.load %0[%arg4, %arg5] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
              %2 = "d2m.tile_exp"(%1) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              affine.store %2, %0[%arg4, %arg5] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
            }
          }
          d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
  // CHECK-LABEL: func.func @test_full_load_store
  // CHECK-NOT: d2m.remote_load
  // CHECK-NOT: d2m.remote_store
  func.func @test_full_load_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream_1 = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream_1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)  {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          // CHECK: %[[LOAD_MEMREF:.*]] = d2m.reserve %{{.*}}
          // CHECK: %[[LOAD_TX:.*]] = d2m.dma_read %{{.*}}[%{{.*}}, %{{.*}}], %[[LOAD_MEMREF]], <0>
          // CHECK-NEXT: d2m.dma_wait %[[LOAD_TX]]
          // CHECK-NEXT: d2m.push %{{.*}}
          d2m.remote_load %stream[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // CHECK: %[[STORE_MEMREF:.*]] = d2m.wait %{{.*}}
          // CHECK: %[[STORE_TX:.*]] = d2m.dma_write %[[STORE_MEMREF]], %{{.*}}[%{{.*}}, %{{.*}}], <0>
          // CHECK-NEXT: d2m.dma_wait %[[STORE_TX]]
          // CHECK-NEXT: d2m.pop %{{.*}}
          d2m.remote_store %stream_1[%0, %1] from %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %1 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%1 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %2 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %2 : !ttcore.tile<32x32, f32>
          }
          d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
  // CHECK-LABEL: func.func @test_local_only
  // CHECK-NOT: d2m.remote_load
  // CHECK-NOT: d2m.remote_store
  func.func @test_local_only(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0:
    %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          %1 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%1 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%0 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %2 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %2 : !ttcore.tile<32x32, f32>
          }
          d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
  // CHECK-LABEL: func.func @test_non_unified_no_change
  // CHECK-NOT: d2m.remote_load
  // CHECK-NOT: d2m.remote_store
  func.func @test_non_unified_no_change(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
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
  // Scatter op lowering: if-guarded mcast DMAWriteOp + semaphore protocol.
  // CHECK-LABEL: func.func @test_scatter_mcast
  func.func @test_scatter_mcast(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // Scatter should lower to:
    //   reserve(outputCb) unconditionally
    //   if (isSender) { wait(inputCb) → dma_write(mcast) → dma_wait → pop(inputCb) → semaphore_set }
    //   else { semaphore_wait }
    //   push(outputCb) unconditionally
    //
    // CHECK: d2m.reserve
    // CHECK: scf.if
    // CHECK:   d2m.wait
    // CHECK:   d2m.dma_write
    // CHECK:   d2m.dma_wait
    // CHECK:   d2m.pop
    // CHECK:   d2m.semaphore_set
    // CHECK: } else {
    // CHECK:   d2m.semaphore_wait
    // CHECK: }
    // CHECK: d2m.push
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%sem0: !d2m.semaphore):
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      d2m.scatter %cb0 into %cb1 mcore[%c0, %c0] mshape[%c1, %c4] {preallocated_semaphores = [0]} : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0(%sem0c: !d2m.semaphore):
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %0 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }
    return
  }

  // Scatter op lowering (unicast): N individual dma_writes + per-core sem_set.
  // CHECK-LABEL: func.func @test_scatter_unicast
  func.func @test_scatter_unicast(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // Unicast scatter should lower to:
    //   reserve(outputCb) unconditionally
    //   if (isSender) {
    //     wait(inputCb)
    //     dma_write to core[0,0]
    //     dma_write to core[0,1]
    //     dma_write to core[1,0]
    //     dma_wait x3
    //     pop(inputCb)
    //     semaphore_set to core[0,1]
    //     semaphore_set to core[1,0]
    //   } else { semaphore_wait }
    //   push(outputCb)
    //
    // CHECK: d2m.reserve
    // CHECK: scf.if
    // CHECK:   d2m.wait
    // CHECK:   d2m.dma_write
    // CHECK:   d2m.dma_write
    // CHECK:   d2m.dma_write
    // CHECK:   d2m.dma_wait
    // CHECK:   d2m.dma_wait
    // CHECK:   d2m.dma_wait
    // CHECK:   d2m.pop
    // CHECK:   d2m.semaphore_set
    // CHECK:   d2m.semaphore_set
    // CHECK: } else {
    // CHECK:   d2m.semaphore_wait
    // CHECK: }
    // CHECK: d2m.push
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%sem0: !d2m.semaphore):
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // 3 cores: sender=(0,0), receivers=(0,1) and (1,0)
      d2m.scatter %cb0 into %cb1 cores[%c0, %c0, %c0, %c1, %c1, %c0] {preallocated_semaphores = [0]} : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }, {
    ^compute0(%sem0c: !d2m.semaphore):
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %0 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }
    return
  }

  // local_copy gets its producer side synchronization inserted.
  // CHECK-LABEL: func.func @test_local_copy
  func.func @test_local_copy() {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {

    // CHECK: %[[SRC:.*]] = d2m.wait
    // CHECK: d2m.reserve
    // CHECK: %[[TX:.*]] = d2m.local_copy %[[SRC]], %{{.*}} indexing_maps
    // CHECK-SAME: -> !d2m.mem_tx
    // CHECK: d2m.dma_wait %[[TX]]
    // CHECK: d2m.push
    // CHECK: d2m.pop
    // CHECK: }, {
    ^datamovement0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb2 = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %src = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          d2m.local_copy %src into %cb2 indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, f32>, #l1> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0:
      %cb2 = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %dst = d2m.wait %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          d2m.pop %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
}
