// RUN: ttmlir-opt --ttcore-register-device --d2m-preallocate-mcast-semaphores -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  // Test 1: RemoteLoadOp with multicast should get preallocated semaphores
  // CHECK-LABEL: func.func @test_mcast_remote_load
  func.func @test_mcast_remote_load(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // Semaphores should be added to both regions (2 per multicast load)
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.cb<{{.*}}>, %[[SEM0:.*]]: !d2m.semaphore, %[[SEM1:.*]]: !d2m.semaphore):
    // CHECK: d2m.remote_load {{.*}} {preallocated_semaphores = [2, 3]}
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.semaphore, %{{.*}}: !d2m.semaphore):
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
          // Multicast remote_load: has mcastStartIndex [%c0, %c0] and mcastShape [%c1, %c4]
          d2m.remote_load %arg0[%0, %1] into %cb0 mcore[%c0, %c0] mshape[%c1, %c4] : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
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

  // Test 2: RemoteLoadOp WITHOUT multicast should NOT get semaphores
  // CHECK-LABEL: func.func @test_non_mcast_remote_load
  func.func @test_non_mcast_remote_load(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // No multicast = no semaphores added
    // Block args should only have CBs, no semaphores
    // CHECK: ^datamovement0(%cb0: !d2m.cb<{{.*}}>, %cb1: !d2m.cb<{{.*}}>):
    // CHECK: d2m.remote_load
    // CHECK-NOT: preallocated_semaphores
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
          // Non-multicast remote_load: no mcast parameters
          d2m.remote_load %arg0[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
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

  // Test 3: Two multicast RemoteLoadOps should each get separate semaphore pairs
  // CHECK-LABEL: func.func @test_two_mcast_remote_loads
  func.func @test_two_mcast_remote_loads(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                         %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // 2 multicast loads = 4 semaphores (2 per load)
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.semaphore, %{{.*}}: !d2m.semaphore, %{{.*}}: !d2m.semaphore, %{{.*}}: !d2m.semaphore):
    // First load gets indices [3, 4]
    // CHECK: d2m.remote_load{{.*}}into %cb0{{.*}} {preallocated_semaphores = [3, 4]}
    // Second load gets indices [5, 6]
    // CHECK: d2m.remote_load{{.*}}into %cb1{{.*}} {preallocated_semaphores = [5, 6]}
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          // First multicast load for operand A
          d2m.remote_load %arg0[%0, %1] into %cb0 mcore[%c0, %c0] mshape[%c1, %c4] : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // Second multicast load for operand B
          d2m.remote_load %arg1[%0, %1] into %cb1 mcore[%c0, %c0] mshape[%c4, %c1] : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %1 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %2 = d2m.reserve %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 4: Mix of multicast and non-multicast loads - only multicast gets semaphores
  // CHECK-LABEL: func.func @test_mixed_mcast_and_non_mcast
  func.func @test_mixed_mcast_and_non_mcast(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                            %arg1: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // Only 1 multicast load = 2 semaphores
    // CHECK: ^{{.*}}(%{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.cb<{{.*}}>, %{{.*}}: !d2m.semaphore, %{{.*}}: !d2m.semaphore):
    // Only the multicast load gets the attribute
    // CHECK: d2m.remote_load{{.*}}into %cb0{{.*}} {preallocated_semaphores = [3, 4]}
    // Non-multicast load does NOT get the attribute
    // CHECK: d2m.remote_load{{.*}}into %cb1{{.*}}{{$}}
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          // Multicast load
          d2m.remote_load %arg0[%0, %1] into %cb0 mcore[%c0, %c0] mshape[%c1, %c4] : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // Non-multicast load
          d2m.remote_load %arg1[%0, %1] into %cb1 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %1 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %2 = d2m.reserve %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 5: Generic with no remote loads at all - nothing happens
  // CHECK-LABEL: func.func @test_no_remote_loads
  func.func @test_no_remote_loads(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // No remote loads = no semaphores added
    // Block args should only have CBs, no semaphores
    // CHECK: ^datamovement0(%cb0: !d2m.cb<{{.*}}>, %cb1: !d2m.cb<{{.*}}>):
    // CHECK-NOT: !d2m.semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // Empty datamovement region
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
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
