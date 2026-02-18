// RUN: ttmlir-opt --ttcore-register-device --d2m-schedule-dma -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {
  // Test 1: Two remote_loads for different CBs - should split into 2 DMA threads
  // CHECK-LABEL: func.func @test_two_remote_loads
  func.func @test_two_remote_loads(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                   %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // CHECK: ^datamovement0
      // CHECK:   d2m.remote_load{{.*}}into %cb0
      // CHECK-NOT: into %cb1
      // CHECK: ^datamovement1
      // CHECK:   d2m.remote_load{{.*}}into %cb1
      // CHECK-NOT: into %cb0
      // CHECK: ^compute0
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          d2m.remote_load %arg0[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_load %arg1[%0, %1] into %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
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

  // Test 2: Single remote_load - should NOT split (only 1 CB with work)
  // CHECK-LABEL: func.func @test_single_remote_load
  func.func @test_single_remote_load(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // Should remain 1 datamovement + 1 compute since only 1 CB has DMA work
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // CHECK-NOT: threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // CHECK: ^datamovement0
      // CHECK: d2m.remote_load{{.*}}into %cb0
      // CHECK: ^compute0
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
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

  // Test 3: Empty datamovement region - should NOT split
  // CHECK-LABEL: func.func @test_empty_datamovement
  func.func @test_empty_datamovement(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // CHECK: ^datamovement0
      // CHECK-NOT: d2m.remote_load
      // CHECK-NOT: d2m.remote_store
      // CHECK: ^compute0
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

  // Test 4: Three remote inputs - should balance across 2 HW threads
  // CHECK-LABEL: func.func @test_three_remote_loads
  func.func @test_three_remote_loads(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                     %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                     %arg2: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // Limited to 2 HW datamovement threads even with 3 CBs
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1, %arg2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // CHECK: ^datamovement0
      // CHECK: ^datamovement1
      // CHECK: ^compute0
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          d2m.remote_load %arg0[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_load %arg1[%0, %1] into %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_load %arg2[%0, %1] into %cb2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %1 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %2 = d2m.wait %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %3 = d2m.reserve %cb3 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 5: Remote load and remote store on different CBs - should split
  // CHECK-LABEL: func.func @test_remote_load_and_store
  func.func @test_remote_load_and_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                        %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // CHECK: ^datamovement0
      // CHECK: ^datamovement1
      // CHECK: ^compute0
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          d2m.remote_load %arg0[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_store %arg1[%0, %1] from %cb2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %1 = d2m.reserve %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 6: Two remote loads + one remote store - verify loads split into different threads
  // CHECK-LABEL: func.func @test_two_loads_one_store
  func.func @test_two_loads_one_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                      %arg2: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // 3 CBs with DMA work -> split into 2 HW threads
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1, %alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%arg2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // With greedy load balancing: cb0 (1 op) -> thread0, cb1 (1 op) -> thread1, cb3 (1 op) -> thread0
      // Thread0 gets cb0 + cb3, Thread1 gets cb1
      // CHECK: ^datamovement0
      // CHECK: d2m.remote_load{{.*}}into %cb0
      // CHECK: d2m.remote_store{{.*}}from %cb3
      // CHECK: ^datamovement1
      // CHECK: d2m.remote_load{{.*}}into %cb1
      // CHECK-NOT: into %cb0
      // CHECK-NOT: d2m.remote_store
      // CHECK: ^compute0
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          d2m.remote_load %arg0[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_load %arg1[%0, %1] into %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_store %arg2[%0, %1] from %cb3 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %1 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %2 = d2m.reserve %cb3 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 7: Single remote store only - should NOT split (only 1 CB with work)
  // CHECK-LABEL: func.func @test_single_remote_store
  func.func @test_single_remote_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                                      %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {

    // CHECK: d2m.generic
    // Should remain 1 datamovement + 1 compute since only 1 CB has DMA work
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
    // CHECK-NOT: threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // CHECK: ^datamovement0
      // CHECK: d2m.remote_store
      // CHECK-NOT: d2m.remote_load
      // CHECK: ^compute0
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          d2m.remote_store %arg1[%0, %1] from %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %1 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 8: Three remote loads + one remote store - verify balanced distribution
  // With 4 CBs each having 1 DMA op, greedy balancing distributes 2 CBs per thread
  // CHECK-LABEL: func.func @test_three_loads_one_store_balanced
  func.func @test_three_loads_one_store_balanced(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                                  %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                                  %arg2: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>,
                                                  %arg3: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>) {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: d2m.generic
    // 4 CBs with DMA work -> split into 2 HW threads, balanced 2 ops each
    // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1, %arg2, %alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%arg3 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb4: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      // With greedy assignment (all ops have count 1):
      // cb0 -> thread0, cb4 -> thread0, cb1 -> thread1, cb2 -> thread1
      // Thread0 gets cb0 + cb4, Thread1 gets cb1 + cb2
      // CHECK: ^datamovement0
      // Thread 0 should have 2 operations: cb0 load and cb4 store
      // CHECK: d2m.remote_load{{.*}}into %cb0
      // CHECK: d2m.remote_store{{.*}}from %cb4
      // CHECK: ^datamovement1
      // Thread 1 should have 2 operations: cb1 load and cb2 load
      // CHECK: d2m.remote_load{{.*}}into %cb1
      // CHECK: d2m.remote_load{{.*}}into %cb2
      // CHECK: ^compute0
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg4 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg5 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg4 : index
          %1 = arith.addi %core1, %arg5 : index
          d2m.remote_load %arg0[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_load %arg1[%0, %1] into %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_load %arg2[%0, %1] into %cb2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.remote_store %arg3[%0, %1] from %cb4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }, {
    ^compute0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb4: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg4 = %c0 to %c1 step %c1 {
        scf.for %arg5 = %c0 to %c1 step %c1 {
          %0 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %1 = d2m.wait %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %2 = d2m.wait %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %3 = d2m.reserve %cb4 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }
}
