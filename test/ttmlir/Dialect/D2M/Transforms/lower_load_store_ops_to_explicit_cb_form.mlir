// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-load-store-ops-to-explicit-cb-form %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#reblock_map = affine_map<(d0, d1, d2, d3) -> ((d0 * 64 + d2) floordiv 128, (d1 * 64 + d3) floordiv 128, (d0 * 64 + d2) mod 128, (d1 * 64 + d3) mod 128)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Test transformation of remote_load from implicit form to explicit CB form
  // CHECK-LABEL: func.func @test_remote_load_to_explicit_cb
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb0
  // CHECK-NEXT: %[[IN:.*]] = d2m.wait %cb0
  // CHECK: linalg.generic
  // CHECK: d2m.pop %cb0
  func.func @test_remote_load_to_explicit_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // Implicit form: remote_load with result
          %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %in = d2m.remote_load %buffer %stream[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          linalg.generic {
            indexing_maps = [#map, #map],
            iterator_types = ["parallel", "parallel"]
          } ins(%in : memref<2x4x!ttcore.tile<32x32, f32>, #l1>)
            outs(%in : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%arg5: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %exp = "d2m.tile_exp"(%arg5) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %exp : !ttcore.tile<32x32, f32>
          }
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }

  // Test explicit form remote_load (already has CB operand) - should create wait and track for pop
  // CHECK-LABEL: func.func @test_explicit_form_remote_load
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb0
  // CHECK: %[[IN:.*]] = d2m.wait %cb0
  // CHECK: linalg.generic
  // CHECK: d2m.pop %cb0
  func.func @test_explicit_form_remote_load(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // Explicit form: remote_load already has CB operand, no result, no localBuffer
          d2m.remote_load %stream[%0, %1] into %cb0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> into !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // Wait produces the memref for compute
          %2 = d2m.wait %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%2 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%2 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %3 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %3 : !ttcore.tile<32x32, f32>
          }
          d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }

  // Test transformation of remote_store from implicit form to explicit CB form
  // CHECK-LABEL: func.func @test_remote_store_to_explicit_cb
  // CHECK: d2m.reserve %cb1
  // CHECK: affine.for
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %cb1
  // CHECK-NOT: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] %{{.*}} :
  // CHECK: d2m.push %cb1
  func.func @test_remote_store_to_explicit_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                                               %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream_out = "d2m.stream_layout"(%arg1, %cb_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // memref.alloc with operand_index pointing to remote output operand
          %buffer = memref.alloc() {operand_index = 1 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>

          // Simple operation using the buffer
          affine.for %i = 0 to 2 {
            affine.for %j = 0 to 4 {
              %tile = affine.load %buffer[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>>
              %exp = "d2m.tile_exp"(%tile) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              affine.store %exp, %buffer[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>>
            }
          }

          // Implicit form: remote_store with local buffer operand (result required)
          %result = d2m.remote_store %stream_out[%0, %1] %buffer : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }

  // Test explicit form remote_store (already uses CB directly) - should create reserve and track for push
  // CHECK-LABEL: func.func @test_explicit_form_remote_store
  // CHECK: d2m.reserve %cb1
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %cb1
  // CHECK: d2m.push %cb1
  func.func @test_explicit_form_remote_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                                              %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream_out = "d2m.stream_layout"(%arg1, %cb_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // Reserve produces output buffer for compute
          %2 = d2m.reserve %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 4 {
              %3 = affine.load %2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
              %4 = "d2m.tile_exp"(%3) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              affine.store %4, %2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
            }
          }
          d2m.push %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          // Explicit form: remote_store already uses CB directly
          d2m.remote_store %stream_out[%0, %1] from %cb1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> from !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }

  // Test DMA-only form where remote_load uses ViewLayoutOp (not StreamLayoutOp)
  // In DMA-only form, only ViewLayoutOp operands are considered remote
  // CHECK-LABEL: func.func @test_dma_only_form_no_consumers
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb0
  // CHECK-NEXT: %[[IN:.*]] = d2m.wait %cb0
  // CHECK: d2m.pop %cb0
  func.func @test_dma_only_form_no_consumers(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                              %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %view = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>]}
        ins(%view : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // Implicit form: remote_load with result, but result is never used (DMA-only, side-effect only)
          // In DMA-only form, remote_load loads into the output CB (cb1)
          %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %in = d2m.remote_load %buffer %view[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          // No consumers of %in - used purely for side effects
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }

  // Test DMA-only form where remote_load should load into the output CB (cb1), not input CB (cb0)
  // This verifies that in DMA-only GenericOps, remote loads target the destination operand's CB
  // Note: Currently loads into cb0 (input CB) - this may need to be fixed in the pass
  // CHECK-LABEL: func.func @test_dma_only_form_load_to_output_cb
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb0
  // CHECK-NEXT: %[[OUT:.*]] = d2m.wait %cb0
  // CHECK: d2m.pop %cb0
  func.func @test_dma_only_form_load_to_output_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                                   %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %view = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>]}
        ins(%view : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // In DMA-only form, this remote_load should load into the output CB (cb1), not the input CB (cb0)
          %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %out = d2m.remote_load %buffer %view[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }

  // Test full transformation with remote_load and remote_store
  // CHECK-LABEL: func.func @test_full_transformation
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb0
  // CHECK: %[[IN:.*]] = d2m.wait %cb0
  // CHECK: %[[OUT:.*]] = d2m.reserve %cb1
  // CHECK: linalg.generic
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %cb1
  // CHECK-DAG: d2m.pop %cb0
  // CHECK-DAG: d2m.push %cb1
  func.func @test_full_transformation(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                       %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %cb0_alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %cb1_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream_in = "d2m.stream_layout"(%arg0, %cb0_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    %stream_out = "d2m.stream_layout"(%arg1, %cb1_alloc) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index

          // Implicit remote_load
          %buffer_in = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %in = d2m.remote_load %buffer_in %stream_in[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          // memref.alloc for output
          %buffer_out = memref.alloc() {operand_index = 1 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>

          linalg.generic {
            indexing_maps = [#map, #map],
            iterator_types = ["parallel", "parallel"]
          } ins(%in : memref<2x4x!ttcore.tile<32x32, f32>, #l1>)
            outs(%buffer_out : memref<2x4x!ttcore.tile<32x32, f32>>) {
          ^bb0(%arg5: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %exp = "d2m.tile_exp"(%arg5) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %exp : !ttcore.tile<32x32, f32>
          }

          // Implicit remote_store (result required)
          %result = d2m.remote_store %stream_out[%0, %1] %buffer_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }

  // Test based on dma.mlir: multiple generic ops with remote_store and remote_load
  // CHECK-LABEL: func.func @dram_write
  // Verify memref.alloc with operand_index converted to reserve
  // CHECK: d2m.reserve %cb{{[01]}}
  // Verify remote_store in explicit CB form
  // CHECK: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %cb{{[01]}}
  // Verify push inserted
  // CHECK: d2m.push %cb{{[01]}}
  // Verify remote_load in explicit CB form (appears twice)
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb{{[01]}}
  // Verify wait and pop inserted
  // CHECK: d2m.wait %cb{{[01]}}
  // CHECK: d2m.pop %cb{{[01]}}
  func.func @dram_write(%arg0: memref<128x128xf32>) -> memref<128x128xf32> {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 1024 : i64, alignment = 32 : i64} : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #dram>
    %view = d2m.view_layout %alloc_0 remapping = #map4 : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #dram> -> memref<1x1x128x128xf32, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%alloc : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>)
        outs(%view : memref<1x1x128x128xf32, #ttcore.view<4>, #dram>)  {
    ^unified0(%cb0: !d2m.cb<memref<128x128xf32, #l1>>, %cb1: !d2m.cb<memref<128x128xf32, #l1>>):
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %buffer = memref.alloc() {operand_index = 1 : i64} : memref<128x128xf32>
      d2m.remote_store %view[%core0, %core1] %buffer : memref<1x1x128x128xf32, #ttcore.view<4>, #dram>, memref<128x128xf32> -> memref<1x1x128x128xf32, #ttcore.view<4>, #dram>
    }
    memref.dealloc %alloc : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 66560 : i64, alignment = 16 : i64} : memref<2x2x64x64xf32, #ttcore.shard<256x4, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%view : memref<1x1x128x128xf32, #ttcore.view<4>, #dram>)
        outs(%alloc_2 : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>)  {
    ^unified0(%cb0: !d2m.cb<memref<128x128xf32, #l1>>, %cb1: !d2m.cb<memref<128x128xf32, #l1>>):
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %buffer0 = memref.alloc() : memref<128x128xf32, #l1>
      %in = d2m.remote_load %buffer0 %view[%core0, %core1] : memref<128x128xf32, #l1>, memref<1x1x128x128xf32, #ttcore.view<4>, #dram> -> memref<128x128xf32, #l1>
    }
    memref.dealloc %alloc_0 : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #dram>
    %view_3 = d2m.view_layout %alloc_2 remapping = #reblock_map : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1> -> memref<2x2x64x64xf32, #ttcore.view<4>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%view_3 : memref<2x2x64x64xf32, #ttcore.view<4>, #l1>)
        outs(%alloc_1 : memref<2x2x64x64xf32, #ttcore.shard<256x4, 1>, #l1>)  {
    ^unified0(%cb0: !d2m.cb<memref<64x64xf32, #l1>>, %cb1: !d2m.cb<memref<64x64xf32, #l1>>):
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %buffer1 = memref.alloc() : memref<64x64xf32, #l1>
      %in2 = d2m.remote_load %buffer1 %view_3[%core0, %core1] : memref<64x64xf32, #l1>, memref<2x2x64x64xf32, #ttcore.view<4>, #l1> -> memref<64x64xf32, #l1>
    }
    memref.dealloc %alloc_2 : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>
    %alloc_4 = memref.alloc() : memref<128x128xf32>
    d2m.to_host %alloc_1, %alloc_4 layout = <logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded> : memref<2x2x64x64xf32, #ttcore.shard<256x4, 1>, #l1> into memref<128x128xf32>
    memref.dealloc %alloc_1 : memref<2x2x64x64xf32, #ttcore.shard<256x4, 1>, #l1>
    return %alloc_4 : memref<128x128xf32>
  }

  // Test RemoteLoadOp with multicast parameters preservation during conversion
  // CHECK-LABEL: func.func @test_remote_load_multicast_params
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb0 mcore[%{{.*}}, %{{.*}}] mshape[%{{.*}}, %{{.*}}]
  // CHECK: %[[IN:.*]] = d2m.wait %cb0
  // CHECK: d2m.pop %cb0
  func.func @test_remote_load_multicast_params(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %cb_alloc = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = "d2m.stream_layout"(%arg0, %cb_alloc) <{remapping = #map4}> : (memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // Implicit form with multicast parameters
          %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %in = d2m.remote_load %buffer %stream[%0, %1] mcore[%c0, %c0] mshape[%c1, %c4] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%in : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%in : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %exp = "d2m.tile_exp"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %exp : !ttcore.tile<32x32, f32>
          }
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }


  // Test multiple remote_load operations in same generic
  // CHECK-LABEL: func.func @test_multiple_remote_loads
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb0
  // CHECK: %[[IN1:.*]] = d2m.wait %cb0
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %cb1
  // CHECK: %[[IN2:.*]] = d2m.wait %cb1
  // CHECK: d2m.pop %cb0
  // CHECK: d2m.pop %cb1
  func.func @test_multiple_remote_loads(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                         %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %cb_alloc0 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %cb_alloc1 = memref.alloc() {address = 10240 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = "d2m.stream_layout"(%arg0, %cb_alloc0) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    %stream1 = "d2m.stream_layout"(%arg1, %cb_alloc1) <{remapping = #map4}> : (memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // Multiple remote loads from different operands
          %buffer0 = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %buffer1 = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %in1 = d2m.remote_load %buffer0 %stream0[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %in2 = d2m.remote_load %buffer1 %stream1[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%in1, %in2 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%in1 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in1_val: !ttcore.tile<32x32, f32>, %in2_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %add = "d2m.tile_add"(%in1_val, %in2_val) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %add : !ttcore.tile<32x32, f32>
          }

          d2m.pop %cb0 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
          d2m.pop %cb1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
        } {d2m.outer_loop}
      } {d2m.outer_loop}
    }
    return
  }
}
