// RUN: ttmlir-opt --d2m-split-unified-thread-v2 %s | FileCheck %s

// Representative lowered (memref/tile) form the real d2m-be-pipeline feeds the
// pass: an elementwise add whose to_layout produces aliased tilize/untilize
// generics, a streaming input CB, an aliased output CB, and dst-register
// compute. Exercises the V2 dataflow CB-op insertion.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#dst = #ttcore.memory_space<dst>
#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>
module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  // CHECK-LABEL: func.func @main
  //
  // Tilize generic: aliased load + aliased store -> split into 2 regions, full
  // reserve/push/wait/pop cycle on each aliased CB; no compute in datamovement.
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  // CHECK: d2m.remote_load %{{.*}} into %{{.*}}
  // CHECK: d2m.remote_store %{{.*}} from %{{.*}}
  // CHECK-NOT: tile_tilize_block
  // CHECK: }, {
  // CHECK: d2m.reserve
  // CHECK: d2m.push
  // CHECK: d2m.wait
  // CHECK: d2m.reserve
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.push
  // CHECK: d2m.wait
  // CHECK: d2m.pop
  // CHECK: d2m.pop
  func.func @main(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: index, %arg3: index) -> memref<64x64xf32> {
    %alloc = memref.alloc() {address = 120224 : i64, alignment = 16 : i64} : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 124320 : i64, alignment = 16 : i64} : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_1 = memref.alloc() {address = 103840 : i64, alignment = 16 : i64} : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    d2m.to_device %arg0, %alloc_1 layout = #layout : memref<64x64xf32> into memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %0 = d2m.operand_alias %alloc_1 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1> -> memref<32x32xf32, #l1>
    %1 = d2m.operand_alias %alloc_0 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_1 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>)
        outs(%alloc_0 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%0, %1 : memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
     {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      d2m.remote_load %0 %alloc_1[%core0, %core1] : memref<32x32xf32, #l1>, memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
      "d2m.tile_tilize_block"(%0, %1) : (memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> ()
      d2m.remote_store %alloc_0[%core0, %core1] %1 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    memref.dealloc %alloc_1 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %alloc_2 = memref.alloc() {address = 128416 : i64, alignment = 16 : i64} : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_3 = memref.alloc() {address = 103840 : i64, alignment = 16 : i64} : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    d2m.to_device %arg1, %alloc_3 layout = #layout : memref<64x64xf32> into memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %2 = d2m.operand_alias %alloc_3 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1> -> memref<32x32xf32, #l1>
    %3 = d2m.operand_alias %alloc_2 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_3 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>)
        outs(%alloc_2 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%2, %3 : memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
     {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      d2m.remote_load %2 %alloc_3[%core0, %core1] : memref<32x32xf32, #l1>, memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
      "d2m.tile_tilize_block"(%2, %3) : (memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> ()
      d2m.remote_store %alloc_2[%core0, %core1] %3 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    memref.dealloc %alloc_3 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %alloc_4 = memref.alloc() {address = 103840 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %alloc_5 = memref.alloc() {address = 112032 : i64, alignment = 16 : i64, d2m.synchronized_buffer = 2 : i32} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %4 = d2m.operand_alias %alloc : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_0, %alloc_2 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%alloc : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%arg2, %arg3, %alloc_4, %alloc_5, %4 : index, index, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>)
     {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %core0 = d2m.core_index(0) : index
      %7 = arith.muli %core0, %arg2 : index
      %core1 = d2m.core_index(1) : index
      %8 = arith.muli %core1, %arg3 : index
      scf.for %arg4 = %c0 to %arg2 step %c1 {
        scf.for %arg5 = %c0 to %arg3 step %c1 {
          %9 = arith.addi %7, %arg4 : index
          %10 = arith.addi %8, %arg5 : index
          d2m.remote_load %alloc_4 %alloc_0[%9, %10] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
          %collapse_shape = memref.collapse_shape %alloc_4 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1> into memref<1x!ttcore.tile<32x32, f32>, #l1>
          %11 = arith.addi %7, %arg4 : index
          %12 = arith.addi %8, %arg5 : index
          d2m.remote_load %alloc_5 %alloc_2[%11, %12] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
          %collapse_shape_8 = memref.collapse_shape %alloc_5 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1> into memref<1x!ttcore.tile<32x32, f32>, #l1>
          %13 = arith.addi %7, %arg4 : index
          %14 = arith.addi %8, %arg5 : index
          %collapse_shape_9 = memref.collapse_shape %4 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #l1> into memref<1x!ttcore.tile<32x32, f32>, #l1>
          scf.for %arg6 = %c0 to %c1 step %c1 {
            scf.for %arg7 = %c0 to %c1 step %c1 {
              %dst = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
              scf.for %arg8 = %c0 to %c1 step %c1 {
                scf.for %arg9 = %c0 to %c1 step %c1 {
                  %15 = arith.addi %arg6, %arg8 : index
                  %16 = arith.addi %arg7, %arg9 : index
                  %17 = arith.addi %15, %16 : index
                  %18 = memref.load %collapse_shape[%17] : memref<1x!ttcore.tile<32x32, f32>, #l1>
                  %19 = arith.addi %arg8, %arg9 : index
                  memref.store %18, %dst[%19] : memref<4x!ttcore.tile<32x32, f32>, #dst>
                }
              }
              scf.for %arg8 = %c0 to %c1 step %c1 {
                scf.for %arg9 = %c0 to %c1 step %c1 {
                  %15 = arith.addi %arg6, %arg8 : index
                  %16 = arith.addi %arg7, %arg9 : index
                  %17 = arith.addi %15, %16 : index
                  %18 = memref.load %collapse_shape_8[%17] : memref<1x!ttcore.tile<32x32, f32>, #l1>
                  %19 = arith.addi %arg8, %arg9 : index
                  %20 = arith.addi %19, %c1 : index
                  memref.store %18, %dst[%20] : memref<4x!ttcore.tile<32x32, f32>, #dst>
                }
              }
              scf.for %arg8 = %c0 to %c1 step %c1 {
                scf.for %arg9 = %c0 to %c1 step %c1 {
                  %15 = arith.addi %arg8, %arg9 : index
                  %16 = arith.addi %15, %c1 : index
                  %17 = memref.load %dst[%16] : memref<4x!ttcore.tile<32x32, f32>, #dst>
                  %18 = arith.addi %arg8, %arg9 : index
                  %19 = memref.load %dst[%18] : memref<4x!ttcore.tile<32x32, f32>, #dst>
                  %20 = "d2m.tile_add"(%19, %17) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
                  %21 = arith.addi %arg8, %arg9 : index
                  memref.store %20, %dst[%21] : memref<4x!ttcore.tile<32x32, f32>, #dst>
                }
              }
              scf.for %arg8 = %c0 to %c1 step %c1 {
                scf.for %arg9 = %c0 to %c1 step %c1 {
                  %15 = arith.addi %arg8, %arg9 : index
                  %16 = memref.load %dst[%15] : memref<4x!ttcore.tile<32x32, f32>, #dst>
                  %17 = arith.addi %arg6, %arg8 : index
                  %18 = arith.addi %arg7, %arg9 : index
                  %19 = arith.addi %17, %18 : index
                  memref.store %16, %collapse_shape_9[%19] : memref<1x!ttcore.tile<32x32, f32>, #l1>
                }
              }
            }
          }
          d2m.remote_store %alloc[%13, %14] %4 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>
        }
      }
    }
    memref.dealloc %alloc_5 : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    memref.dealloc %alloc_4 : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    memref.dealloc %alloc_0 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    memref.dealloc %alloc_2 : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_6 = memref.alloc() : memref<64x64xf32>
    %alloc_7 = memref.alloc() {address = 103840 : i64, alignment = 16 : i64} : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %5 = d2m.operand_alias %alloc : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    %6 = d2m.operand_alias %alloc_7 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1> -> memref<32x32xf32, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%alloc_7 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>)
        additionalArgs(%5, %6 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<32x32xf32, #l1>)
     {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      d2m.remote_load %5 %alloc[%core0, %core1] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
      "d2m.tile_untilize_block"(%5, %6) : (memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<32x32xf32, #l1>) -> ()
      d2m.remote_store %alloc_7[%core0, %core1] %6 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>, memref<32x32xf32, #l1>
    }
    memref.dealloc %alloc : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.to_host %alloc_7, %alloc_6 layout = #layout : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1> into memref<64x64xf32>
    memref.dealloc %alloc_7 : memref<2x2x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    return %alloc_6 : memref<64x64xf32>
  }
}
