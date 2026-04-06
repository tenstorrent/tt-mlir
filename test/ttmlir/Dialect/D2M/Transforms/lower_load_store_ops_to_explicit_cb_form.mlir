// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-load-store-ops-to-explicit-cb-form %s | FileCheck %s

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#reblock_map = affine_map<(d0, d1, d2, d3) -> ((d0 * 64 + d2) floordiv 128, (d1 * 64 + d3) floordiv 128, (d0 * 64 + d2) mod 128, (d1 * 64 + d3) mod 128)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Test transformation of remote_load from implicit form to explicit CB form
  // CHECK-LABEL: func.func @test_remote_load_to_explicit_cb
  // CHECK: %[[CB0:.*]] = d2m.get_cb(0)
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[CB0]]
  // CHECK-NEXT: %[[IN:.*]] = d2m.wait %[[CB0]]
  // CHECK: linalg.generic
  // CHECK: d2m.pop %[[CB0]]
  func.func @test_remote_load_to_explicit_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0:
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
          d2m.remote_load %buffer %stream[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          linalg.generic {
            indexing_maps = [#map, #map],
            iterator_types = ["parallel", "parallel"]
          } ins(%buffer : memref<2x4x!ttcore.tile<32x32, f32>, #l1>)
            outs(%buffer : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%arg5: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %exp = "d2m.tile_exp"(%arg5) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %exp : !ttcore.tile<32x32, f32>
          }
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test transformation of remote_store from implicit form to explicit CB form
  // The pass converts implicit remote_store to explicit CB form with reserve/store from CB/push.
  // CHECK-LABEL: func.func @test_remote_store_to_explicit_cb
  // CHECK: %[[CB1:.*]] = d2m.get_cb(1)
  // CHECK: d2m.reserve %[[CB1]]
  // CHECK: affine.for
  // CHECK: d2m.push %[[CB1]]
  // CHECK-NEXT: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[CB1]]
  func.func @test_remote_store_to_explicit_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                                               %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>) {
    ^unified0:
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
          d2m.remote_store %stream_out[%0, %1] %buffer : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test implicit remote_store fed by local memref.alloc buffer.
  // CHECK-LABEL: func.func @test_remote_store_from_reserved_buffer
  // CHECK: d2m.reserve %[[CB1:[0-9]+]]
  // CHECK: d2m.push %[[CB1]]
  // CHECK-NEXT: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[CB1]]
  func.func @test_remote_store_from_reserved_buffer(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                                                     %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index
          // Local working buffer for compute before remote_store conversion.
          %2 = memref.alloc() {operand_index = 1 : i64} : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 4 {
              %3 = affine.load %2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
              %4 = "d2m.tile_exp"(%3) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              affine.store %4, %2[%arg5, %arg6] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
            }
          }
          // Implicit form store to exercise conversion to explicit CB store.
          d2m.remote_store %arg1[%0, %1] %2 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // External allocs carried via additionalArgs must remain outside the
  // generic; only in-body uses are rewritten to reserve.
  // CHECK-LABEL: func.func @test_remote_store_keeps_external_alloc
  // CHECK: %[[EXT:.*]] = memref.alloc()
  // CHECK: d2m.generic
  // CHECK: additionalArgs(%[[EXT]] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>)
  // CHECK: %[[CB1:.*]] = d2m.get_cb(1)
  // CHECK-NEXT: %[[RES:.*]] = d2m.reserve %[[CB1]]
  // CHECK: affine.load %[[RES]]
  // CHECK: affine.store %{{.*}}, %[[RES]]
  // CHECK: d2m.push %[[CB1]]
  // CHECK-NEXT: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[CB1]]
  // CHECK: memref.dealloc %[[EXT]]
  func.func @test_remote_store_keeps_external_alloc(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                                                    %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    %external = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>

    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        additionalArgs(%external : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 4 {
          %tile = affine.load %external[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %exp = "d2m.tile_exp"(%tile) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %exp, %external[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        }
      }
      d2m.remote_store %stream_out[%core0, %core1] %external : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    }
    memref.dealloc %external : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    return
  }

  // Test DMA-only form where remote_load should load into the output CB (cb1), not input CB (cb0)
  // This verifies that in DMA-only GenericOps, remote loads target the destination operand's CB
  // Note: Currently loads into cb0 (input CB) - this may need to be fixed in the pass
  // CHECK-LABEL: func.func @test_dma_only_form_load_to_output_cb
  // CHECK: %[[CB0:.*]] = d2m.get_cb(0)
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[CB0]]
  // CHECK-NEXT: %[[OUT:.*]] = d2m.wait %[[CB0]]
  // CHECK: d2m.pop %[[CB0]]
  func.func @test_dma_only_form_load_to_output_cb(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                                   %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %view = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>]}
        ins(%view : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>)
        outs(%arg1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^datamovement0:
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
          d2m.remote_load %buffer %view[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test full transformation with remote_load and remote_store
  // Both remote_load and remote_store are converted to explicit CB form.
  // CHECK-LABEL: func.func @test_full_transformation
  // CHECK-DAG: %[[CB0:.*]] = d2m.get_cb(0)
  // CHECK-DAG: %[[CB1:.*]] = d2m.get_cb(1)
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[CB0]]
  // CHECK-NEXT: %[[IN:.*]] = d2m.wait %[[CB0]]
  // CHECK: d2m.reserve %[[CB1]]
  // CHECK: linalg.generic
  // CHECK: d2m.push %[[CB1]]
  // CHECK-NEXT: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[CB1]]
  // CHECK: d2m.pop %[[CB0]]
  func.func @test_full_transformation(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                       %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream_in = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>) {
    ^unified0:
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
          d2m.remote_load %buffer_in %stream_in[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          // memref.alloc for output
          %buffer_out = memref.alloc() {operand_index = 1 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>

          linalg.generic {
            indexing_maps = [#map, #map],
            iterator_types = ["parallel", "parallel"]
          } ins(%buffer_in : memref<2x4x!ttcore.tile<32x32, f32>, #l1>)
            outs(%buffer_out : memref<2x4x!ttcore.tile<32x32, f32>>) {
          ^bb0(%arg5: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %exp = "d2m.tile_exp"(%arg5) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %exp : !ttcore.tile<32x32, f32>
          }

          // Implicit remote_store (result required)
          d2m.remote_store %stream_out[%0, %1] %buffer_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test based on dma.mlir: multiple generic ops with remote_store and remote_load
  // CHECK-LABEL: func.func @dram_write
  // Verify remote_store converted to explicit CB form
  // CHECK: d2m.reserve %[[CB1:[0-9]+]]
  // CHECK: d2m.push %[[CB1]]
  // CHECK-NEXT: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[CB1]]
  // Verify remote_load in explicit CB form
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[CB0:[0-9]+]]
  // Verify wait and pop inserted
  // CHECK-NEXT: d2m.wait %[[CB0]]
  // CHECK: d2m.pop %[[CB0]]
  func.func @dram_write(%arg0: memref<128x128xf32>) -> memref<128x128xf32> {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 1024 : i64, alignment = 32 : i64} : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #dram>
    %view = d2m.view_layout %alloc_0 remapping = #map4 : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #dram> -> memref<1x1x128x128xf32, #ttcore.view<4>, #dram>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%alloc : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>)
        outs(%view : memref<1x1x128x128xf32, #ttcore.view<4>, #dram>)  {
    ^unified0:
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
    ^unified0:
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %buffer0 = memref.alloc() : memref<128x128xf32, #l1>
      d2m.remote_load %buffer0 %view[%core0, %core1] : memref<128x128xf32, #l1>, memref<1x1x128x128xf32, #ttcore.view<4>, #dram> -> memref<128x128xf32, #l1>
    }
    memref.dealloc %alloc_0 : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #dram>
    %view_3 = d2m.view_layout %alloc_2 remapping = #reblock_map : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1> -> memref<2x2x64x64xf32, #ttcore.view<4>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x2>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%view_3 : memref<2x2x64x64xf32, #ttcore.view<4>, #l1>)
        outs(%alloc_1 : memref<2x2x64x64xf32, #ttcore.shard<256x4, 1>, #l1>)  {
    ^unified0:
      %core0 = d2m.core_index(0) {phys_to_virt_map = affine_map<() -> ()>} : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = affine_map<() -> ()>} : index
      %buffer1 = memref.alloc() : memref<64x64xf32, #l1>
      d2m.remote_load %buffer1 %view_3[%core0, %core1] : memref<64x64xf32, #l1>, memref<2x2x64x64xf32, #ttcore.view<4>, #l1> -> memref<64x64xf32, #l1>
    }
    memref.dealloc %alloc_2 : memref<1x1x128x128xf32, #ttcore.shard<512x4, 1>, #l1>
    %alloc_4 = memref.alloc() : memref<128x128xf32>
    d2m.to_host %alloc_1, %alloc_4 layout = <logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded> : memref<2x2x64x64xf32, #ttcore.shard<256x4, 1>, #l1> into memref<128x128xf32>
    memref.dealloc %alloc_1 : memref<2x2x64x64xf32, #ttcore.shard<256x4, 1>, #l1>
    return %alloc_4 : memref<128x128xf32>
  }

  // Test RemoteLoadOp with multicast parameters preservation during conversion
  // CHECK-LABEL: func.func @test_remote_load_multicast_params
  // CHECK: %[[CB0:.*]] = d2m.get_cb(0)
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[CB0]] mcore[%{{.*}}, %{{.*}}] mshape[%{{.*}}, %{{.*}}]
  // CHECK-NEXT: %[[IN:.*]] = d2m.wait %[[CB0]]
  // CHECK: d2m.pop %[[CB0]]
  func.func @test_remote_load_multicast_params(%arg0: memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0:
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
          %bufferIn = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %bufferIn %stream[%0, %1] mcore[%c0, %c0] mshape[%c1, %c4] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<4x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %bufferOut = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%bufferIn : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%bufferOut : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %exp = "d2m.tile_exp"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %exp : !ttcore.tile<32x32, f32>
          }
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }


  // Test multiple remote_load operations in same generic
  // CHECK-LABEL: func.func @test_multiple_remote_loads
  // CHECK-DAG: %[[CB0:.*]] = d2m.get_cb(0)
  // CHECK-DAG: %[[CB1:.*]] = d2m.get_cb(1)
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[CB0]]
  // CHECK-NEXT: %[[IN1:.*]] = d2m.wait %[[CB0]]
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[CB1]]
  // CHECK-NEXT: %[[IN2:.*]] = d2m.wait %[[CB1]]
  // CHECK: d2m.pop %[[CB0]]
  // CHECK: d2m.pop %[[CB1]]
  func.func @test_multiple_remote_loads(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                         %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream0 = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    %stream1 = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream0, %stream1 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0:
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
          d2m.remote_load %buffer0 %stream0[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          d2m.remote_load %buffer1 %stream1[%0, %1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %bufferOut = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>

          linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
            ins(%buffer0, %buffer1 : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>)
            outs(%bufferOut : memref<2x4x!ttcore.tile<32x32, f32>, #l1>) {
          ^bb0(%in1_val: !ttcore.tile<32x32, f32>, %in2_val: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
            %add = "d2m.tile_add"(%in1_val, %in2_val) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %add : !ttcore.tile<32x32, f32>
          }
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Regression: shared-CB explicit load/store should keep load and store adjacent.
  // CHECK-LABEL: func.func @test_shared_cb_load_store_anchor_order
  // CHECK: d2m.remote_load %{{.*}}[%{{.*}}, %{{.*}}] into %[[LOADCB:[0-9]+]]
  // CHECK-NEXT: d2m.remote_store %{{.*}}[%{{.*}}, %{{.*}}] from %[[LOADCB]]
  func.func @test_shared_cb_load_store_anchor_order(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                                    %arg1: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram>,
                                                    %sem: !d2m.global_semaphore) {
    %stream_in = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>) {
    ^unified0:
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_load %buffer %stream_in[%core0, %core1] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      d2m.remote_store %stream_out[%core0, %core1] %buffer
      : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram>
    }
    return
  }

  // Test: local_copy with scratch intermediate dst.
  // The local_copy dst traces back to a d2m.reserve, so the pass leaves it in
  // memref form: no new CB, no wait, no pop for the scratch slot.
  // CHECK-LABEL: func.func @test_local_copy_scratch_dst
  func.func @test_local_copy_scratch_dst(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    %scratch_buf = memref.alloc() : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #l1>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>], scratch_inputs = array<i64: 2>}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #l1>) {
    // CHECK-DAG: %[[CB0:.*]] = d2m.get_cb(0)
    // CHECK-DAG: %[[SCRATCH_CB:.*]] = d2m.get_cb(2)
    // Scratch local_copy is converted to explicit CB form using the scratch CB.
    // CHECK: d2m.remote_load {{.*}} into %[[CB0]]
    // CHECK-NEXT: %[[IN:.*]] = d2m.wait %[[CB0]]
    // CHECK: d2m.local_copy %[[IN]] into %[[SCRATCH_CB]] indexing_maps
    // CHECK-NOT: d2m.get_cb(3)
    // No pop for CB0: local_copy (DM op) is the only consumer of %IN.
    // CHECK-NOT: d2m.pop %[[CB0]]
    // Pop for scratch CB: wait result has no users, slot must be released.
    // CHECK: d2m.pop %[[SCRATCH_CB]]
    ^unified0:
      %scratch_cb = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>>
      %scratch = d2m.reserve %scratch_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
      %mid = memref.subview %scratch[0, 0][2, 4][1, 1] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1> to memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index

          %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
          d2m.remote_load %buffer %stream[%0, %1] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

          d2m.local_copy %buffer, %mid indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // CHECK-LABEL: func.func @test_local_copy_chain_to_compute
  func.func @test_local_copy_chain_to_compute(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %alloc = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #l1>
    %stream = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%alloc : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #l1>) {
    // CHECK-DAG: %[[CB0:.*]] = d2m.get_cb(0)
    // CHECK-DAG: %[[SCRATCH_CB2:.*]] = d2m.get_cb(2)
    // CHECK-DAG: %[[SCRATCH_CB3:.*]] = d2m.get_cb(3)
    // All ops converted to explicit CB form.
    // CHECK: d2m.remote_load {{.*}} into %[[CB0]]
    // CHECK-NEXT: %[[IN:.*]] = d2m.wait %[[CB0]]
    // CHECK: d2m.local_copy %[[IN]] into %[[SCRATCH_CB2]] indexing_maps
    // CHECK-NEXT: %[[MID:.*]] = d2m.wait %[[SCRATCH_CB2]]
    // CHECK: d2m.local_copy %[[MID]] into %[[SCRATCH_CB3]] indexing_maps
    // CHECK-NEXT: %[[FINAL:.*]] = d2m.wait %[[SCRATCH_CB3]]
    // CHECK: linalg.generic
    // Pop for SCRATCH_CB3: linalg.generic is a compute consumer.
    // CHECK: d2m.pop %[[SCRATCH_CB3]]
    // No pops — CB0's only consumer is local_copy (DM op).
    // CHECK-NOT: d2m.pop
    ^unified0:
      %scratch_cb2 = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>>
      %scratch2 = d2m.reserve %scratch_cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
      %mid = memref.subview %scratch2[0, 0][2, 4][1, 1] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1> to memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

      %scratch_cb3 = d2m.get_cb(3) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>>
      %scratch3 = d2m.reserve %scratch_cb3 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
      %final = memref.subview %scratch3[0, 0][2, 4][1, 1] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1> to memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index

          %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
          d2m.remote_load %buffer %stream[%0, %1] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

          d2m.local_copy %buffer, %mid indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

          d2m.local_copy %mid, %final indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

          linalg.generic {
            indexing_maps = [#map, #map],
            iterator_types = ["parallel", "parallel"]
          } ins(%final : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>)
            outs(%final : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>) {
          ^bb0(%arg5: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
            %exp = "d2m.tile_exp"(%arg5) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            linalg.yield %exp : !ttcore.tile<32x32, bf16>
          }
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Test: chained local_copy → local_copy → remote_store with scratch
  // All consumers are DM ops — no pops are inserted.
  // CHECK-LABEL: func.func @test_local_copy_chain_to_remote_store
  func.func @test_local_copy_chain_to_remote_store(%arg0: memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #dram>,
                                                    %arg1: memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #dram>) {
    %stream_in = d2m.view_layout %arg0 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
    %stream_out = d2m.view_layout %arg1 remapping = #map4 : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #dram> -> memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>

    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%stream_in : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>)
        outs(%stream_out : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>) {

    // CHECK-DAG: %[[CB0:.*]] = d2m.get_cb(0)
    // CHECK-DAG: %[[SCRATCH_CB2:.*]] = d2m.get_cb(2)
    // CHECK-DAG: %[[SCRATCH_CB3:.*]] = d2m.get_cb(3)
    // All ops converted to explicit CB form.
    // CHECK: d2m.remote_load {{.*}} into %[[CB0]]
    // CHECK-NEXT: %[[IN:.*]] = d2m.wait %[[CB0]]
    // CHECK: d2m.local_copy %[[IN]] into %[[SCRATCH_CB2]] indexing_maps
    // CHECK-NEXT: %[[MID:.*]] = d2m.wait %[[SCRATCH_CB2]]
    // CHECK: d2m.local_copy %[[MID]] into %[[SCRATCH_CB3]] indexing_maps
    // remote_store finds scratch CB3 via the wait chain.
    // CHECK: d2m.push %[[SCRATCH_CB3]]
    // CHECK-NEXT: d2m.remote_store {{.*}} from %[[SCRATCH_CB3]]
    // No pops — all consumers are DM ops.
    // CHECK-NOT: d2m.pop
    ^unified0:
      %scratch_cb2 = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>>
      %scratch2 = d2m.reserve %scratch_cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
      %mid = memref.subview %scratch2[0, 0][2, 4][1, 1] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1> to memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

      %scratch_cb3 = d2m.get_cb(3) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>>
      %scratch3 = d2m.reserve %scratch_cb3 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, bf16>, #l1>> -> memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
      %final = memref.subview %scratch3[0, 0][2, 4][1, 1] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1> to memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        %core0 = d2m.core_index(0) : index
        %core1 = d2m.core_index(1) : index
        scf.for %arg4 = %c0 to %c1 step %c1 {
          %0 = arith.addi %core0, %arg3 : index
          %1 = arith.addi %core1, %arg4 : index

          %buffer = memref.alloc() : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>
          d2m.remote_load %buffer %stream_in[%0, %1] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram> -> memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

          d2m.local_copy %buffer, %mid indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

          d2m.local_copy %mid, %final indexing_maps = [#map, #map] : memref<2x4x!ttcore.tile<32x32, bf16>, #l1>, memref<2x4x!ttcore.tile<32x32, bf16>, #l1>

          d2m.remote_store %stream_out[%0, %1] %final : memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>, memref<2x4x!ttcore.tile<32x32, bf16>, #l1> -> memref<2x4x2x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x4096, 1>, #ttcore.view<4>, #dram>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Scatter implicit form → explicit CB form with inputCb + outputCb + wait + pop.
  // src goes in ins (read from), dst goes in outs (written to).
  // CHECK-LABEL: func.func @test_scatter_to_explicit_cb
  func.func @test_scatter_to_explicit_cb(
      %arg0: memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    %alloc_dst = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>

    // Scatter gets converted: src → inputCb, dst → outputCb
    // CHECK: d2m.scatter %{{.*}} into %{{.*}}
    // CHECK: d2m.wait
    // CHECK: d2m.pop
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0 : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%alloc_dst : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^unified0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.for %arg3 = %c0 to %c1 step %c1 {
        scf.for %arg4 = %c0 to %c1 step %c1 {
          d2m.scatter %arg0, %alloc_dst mcore[%c0, %c0] mshape[%c2, %c4] : memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

}
