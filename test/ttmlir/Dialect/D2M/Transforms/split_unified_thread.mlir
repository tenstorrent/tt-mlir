// RUN: ttmlir-opt --ttcore-register-device --d2m-split-unified-thread %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Test 1: Remote load with explicit CB form
  // Verifies:
  // - unified thread splits into datamovement and compute
  // - remote_load and index calculations go to datamovement
  // - wait, pop, and linalg.generic stay in compute
  // CHECK-LABEL: func.func @test_remote_load_explicit_cb
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  func.func @test_remote_load_explicit_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %alloc_0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    // CHECK: ^datamovement0
    // CHECK: scf.for
    // CHECK: d2m.core_index(0)
    // CHECK: d2m.core_index(1)
    // CHECK: scf.for
    // CHECK: arith.addi
    // CHECK: arith.addi
    // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %{{.*}}
    // CHECK-NOT: d2m.wait
    // CHECK-NOT: d2m.pop
    // CHECK-NOT: linalg.generic

    // CHECK: ^compute0
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.wait %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.core_index
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg1 : index
          %1 = arith.addi %core1, %arg2 : index
          // Explicit CB form: remote_load into CB, no result
          d2m.remote_load %stream[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // Wait produces the memref for compute
          %2 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%2 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %3 : !ttcore.tile<32x32, f32>
          }
          d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 2: Remote store with explicit CB form
  // Verifies:
  // - remote_store and index calculations go to datamovement
  // - reserve, push, and compute ops stay in compute
  // CHECK-LABEL: func.func @test_remote_store_explicit_cb
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  func.func @test_remote_store_explicit_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                                            %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg1, %alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    // CHECK: ^datamovement0
    // CHECK: scf.for
    // CHECK: d2m.core_index(0)
    // CHECK: d2m.core_index(1)
    // CHECK: scf.for
    // CHECK: arith.addi
    // CHECK: arith.addi
    // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %{{.*}}
    // CHECK-NOT: d2m.reserve
    // CHECK-NOT: d2m.push
    // CHECK-NOT: affine.for

    // CHECK: ^compute0
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: d2m.tile_exp
    // CHECK: d2m.push %{{.*}}
    // CHECK-NOT: d2m.remote_store
    // CHECK-NOT: d2m.core_index
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          // Reserve produces output buffer for compute
          %2 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          affine.for %arg4 = 0 to 2 {
            affine.for %arg5 = 0 to 4 {
              %3 = affine.load %2[%arg4, %arg5] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
              %4 = "d2m.tile_exp"(%3) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              affine.store %4, %2[%arg4, %arg5] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
            }
          }
          d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // Explicit CB form: remote_store from CB
          d2m.remote_store %stream[%0, %1] from %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 3: Full transformation with both remote_load and remote_store
  // Verifies:
  // - Both remote_load and remote_store go to datamovement
  // - All CB ops (wait, reserve, push, pop) stay in compute
  // - Compute operations (linalg.generic) stay in compute
  // CHECK-LABEL: func.func @test_full_load_store
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  func.func @test_full_load_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                   %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream_in = "d2m.stream_layout"(%arg0, %alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    %stream_out = "d2m.stream_layout"(%arg1, %alloc_0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    // CHECK: ^datamovement0
    // CHECK: scf.for
    // CHECK: d2m.core_index(0)
    // CHECK: d2m.core_index(1)
    // CHECK: scf.for
    // CHECK: arith.addi
    // CHECK: arith.addi
    // CHECK: d2m.remote_load
    // CHECK: d2m.remote_store
    // CHECK-NOT: d2m.wait
    // CHECK-NOT: d2m.reserve
    // CHECK-NOT: d2m.push
    // CHECK-NOT: d2m.pop
    // CHECK-NOT: linalg.generic

    // CHECK: ^compute0
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.wait %{{.*}}
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.push %{{.*}}
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.remote_store
    // CHECK-NOT: d2m.core_index
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg2 : index
          %1 = arith.addi %core1, %arg3 : index
          // Remote load into input CB
          d2m.remote_load %stream_in[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // Wait and reserve for compute
          %2 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %3 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%3 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %4 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %4 : !ttcore.tile<32x32, f32>
          }
          d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // Remote store from output CB
          d2m.remote_store %stream_out[%0, %1] from %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test 4: Local operands only (no remote ops)
  // Verifies:
  // - Unified thread still splits into datamovement and compute
  // - Datamovement region is empty (only loops preserved)
  // - All CB ops stay in compute
  // CHECK-LABEL: func.func @test_local_only
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  func.func @test_local_only(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // CHECK: ^datamovement0
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.remote_store
    // CHECK-NOT: d2m.wait
    // CHECK-NOT: d2m.reserve

    // CHECK: ^compute0
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: d2m.reserve %{{.*}}
    // CHECK: d2m.push %{{.*}}
    // CHECK: d2m.wait %{{.*}}
    // CHECK: linalg.generic
    // CHECK: d2m.pop %{{.*}}
    // CHECK-NOT: d2m.remote_load
    // CHECK-NOT: d2m.remote_store
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          // Local operands: reserve/push for output, wait for input
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

  // Test 5: Non-unified thread (should not be transformed)
  // Verifies:
  // - GenericOp with non-unified threads is left unchanged
  // CHECK-LABEL: func.func @test_non_unified_no_change
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<compute>]
  // CHECK-NOT: #d2m.thread<datamovement>
  func.func @test_non_unified_no_change(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
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
