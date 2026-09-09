// RUN: ttmlir-opt --ttcore-register-device --d2m-split-unified-thread %s | FileCheck %s --implicit-check-not=d2m.synchronized_region

// Lowered-loops regression coverage for unified-thread splitting. Unlike the
// other split tests (which feed the high-level linalg.generic compute form),
// this fixture is captured from the real pipeline immediately before
// d2m-split-unified-thread: the compute body is already lowered to scf.for
// nests with memref.load/store into #dst plus tile_*_block ops. This is the
// form synchronized_region was built to aggregate, so it guards the behavior
// the Pass 3 rewrite must preserve.

// CHECK-LABEL: func.func @maximum
// The unified generics are split into datamovement + compute regions.
// CHECK: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
// Tilize/untilize and the elementwise maximum nest survive on the compute side,
// wrapped by the CB synchronization protocol (reserve/push/wait/pop).
// CHECK-DAG: d2m.tile_tilize_block
// CHECK-DAG: d2m.tile_maximum
// CHECK-DAG: d2m.tile_untilize_block
// CHECK-DAG: d2m.reserve
// CHECK-DAG: d2m.wait
// CHECK-DAG: d2m.push
// CHECK-DAG: d2m.pop
module attributes {ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @maximum(%arg0: memref<64x128xf32>, %arg1: memref<64x128xf32>) -> memref<64x128xf32> attributes {tt.function_type = "forward_device"} {
    %alloc = memref.alloc() {address = 9216 : i64, alignment = 16 : i64} : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_0 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    d2m.to_device %arg0, %alloc_0 layout = <logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded> : memref<64x128xf32> into memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    %0 = d2m.operand_alias %alloc_0 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>> -> memref<32x32xf32, #ttcore.memory_space<l1>>
    %1 = d2m.operand_alias %alloc : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_0 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>)
        outs(%alloc : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
        additionalArgs(%0, %1 : memref<32x32xf32, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      d2m.remote_load %0 %alloc_0[%core0, %core1] : memref<32x32xf32, #ttcore.memory_space<l1>>, memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
      "d2m.tile_tilize_block"(%0, %1) : (memref<32x32xf32, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) -> ()
      d2m.remote_store %alloc[%core0, %core1] %1 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    }
    memref.dealloc %alloc_0 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    %alloc_1 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_2 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    d2m.to_device %arg1, %alloc_2 layout = <logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded> : memref<64x128xf32> into memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    %2 = d2m.operand_alias %alloc_2 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>> -> memref<32x32xf32, #ttcore.memory_space<l1>>
    %3 = d2m.operand_alias %alloc_1 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_2 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>)
        outs(%alloc_1 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
        additionalArgs(%2, %3 : memref<32x32xf32, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      d2m.remote_load %2 %alloc_2[%core0, %core1] : memref<32x32xf32, #ttcore.memory_space<l1>>, memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
      "d2m.tile_tilize_block"(%2, %3) : (memref<32x32xf32, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>) -> ()
      d2m.remote_store %alloc_1[%core0, %core1] %3 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    }
    memref.dealloc %alloc_2 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    %alloc_3 = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %4 = d2m.operand_alias %alloc : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    %5 = d2m.operand_alias %alloc_1 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    %6 = d2m.operand_alias %alloc_3 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc, %alloc_1 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
        outs(%alloc_3 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
        additionalArgs(%4, %5, %6 : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>)
        {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %core0_6 = d2m.core_index(0) : index
      %core1_7 = d2m.core_index(1) : index
      d2m.remote_load %4 %alloc[%core0, %core1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
      %collapse_shape = memref.collapse_shape %4 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      d2m.remote_load %5 %alloc_1[%core0_6, %core1_7] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
      %collapse_shape_8 = memref.collapse_shape %5 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %collapse_shape_9 = memref.collapse_shape %6 [[0, 1]] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> into memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c1 step %c1 {
          %dst = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c1 step %c1 {
              %9 = arith.addi %arg2, %arg4 : index
              %10 = arith.addi %arg3, %arg5 : index
              %11 = arith.addi %9, %10 : index
              %12 = memref.load %collapse_shape[%11] : memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
              %13 = arith.addi %arg4, %arg5 : index
              memref.store %12, %dst[%13] : memref<4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
            }
          }
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c1 step %c1 {
              %9 = arith.addi %arg2, %arg4 : index
              %10 = arith.addi %arg3, %arg5 : index
              %11 = arith.addi %9, %10 : index
              %12 = memref.load %collapse_shape_8[%11] : memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
              %13 = arith.addi %arg4, %arg5 : index
              %14 = arith.addi %13, %c1 : index
              memref.store %12, %dst[%14] : memref<4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
            }
          }
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c1 step %c1 {
              %9 = arith.addi %arg4, %arg5 : index
              %10 = memref.load %dst[%9] : memref<4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
              %11 = arith.addi %arg4, %arg5 : index
              %12 = arith.addi %11, %c1 : index
              %13 = memref.load %dst[%12] : memref<4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
              %14 = "d2m.tile_maximum"(%10, %13) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              %15 = arith.addi %arg4, %arg5 : index
              memref.store %14, %dst[%15] : memref<4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
            }
          }
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c1 step %c1 {
              %9 = arith.addi %arg4, %arg5 : index
              %10 = memref.load %dst[%9] : memref<4x!ttcore.tile<32x32, f32>, #ttcore.memory_space<dst>>
              %11 = arith.addi %arg2, %arg4 : index
              %12 = arith.addi %arg3, %arg5 : index
              %13 = arith.addi %11, %12 : index
              memref.store %10, %collapse_shape_9[%13] : memref<1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
            }
          }
        }
      }
      %core0_10 = d2m.core_index(0) : index
      %core1_11 = d2m.core_index(1) : index
      d2m.remote_store %alloc_3[%core0_10, %core1_11] %6 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    }
    memref.dealloc %alloc : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    memref.dealloc %alloc_1 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    %alloc_4 = memref.alloc() : memref<64x128xf32>
    %alloc_5 = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    %7 = d2m.operand_alias %alloc_3 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
    %8 = d2m.operand_alias %alloc_5 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>> -> memref<32x32xf32, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_3 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>)
        outs(%alloc_5 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>)
        additionalArgs(%7, %8 : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<32x32xf32, #ttcore.memory_space<l1>>)
        {
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      d2m.remote_load %7 %alloc_3[%core0, %core1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
      "d2m.tile_untilize_block"(%7, %8) : (memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<32x32xf32, #ttcore.memory_space<l1>>) -> ()
      d2m.remote_store %alloc_5[%core0, %core1] %8 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>, memref<32x32xf32, #ttcore.memory_space<l1>>
    }
    memref.dealloc %alloc_3 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<l1>>
    d2m.to_host %alloc_5, %alloc_4 layout = <logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded> : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>> into memref<64x128xf32>
    memref.dealloc %alloc_5 : memref<2x4x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    return %alloc_4 : memref<64x128xf32>
  }
}
