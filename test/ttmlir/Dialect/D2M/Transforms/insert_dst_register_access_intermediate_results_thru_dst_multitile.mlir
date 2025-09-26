// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access="use-tile-matmul=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
module attributes {ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 100352, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 2560032, dram_unreserved_end = 1073139712, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_register_size_tiles = 8, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]>
  func.func @eltwise_unary_chain_multi_tile(%arg0: memref<128x128xbf16>) -> memref<128x128xbf16> {
    %alloc = memref.alloc() {address = 133120 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #ttcore.memory_space<l1>>
    %alloc_0 = memref.alloc() {address = 100352 : i64, alignment = 16 : i64} : memref<1x1x128x128xbf16, #ttcore.shard<256x2>, #ttcore.memory_space<l1>>
    d2m.to_layout %arg0, %alloc_0 : memref<128x128xbf16> into memref<1x1x128x128xbf16, #ttcore.shard<256x2>, #ttcore.memory_space<l1>> hostInfo = <logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%alloc_0 : memref<1x1x128x128xbf16, #ttcore.shard<256x2>, #ttcore.memory_space<l1>>)
        outs(%alloc : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<128x128xbf16, #ttcore.memory_space<l1>>, %cb1: memref<4x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>):
      "d2m.tile_tilize_block"(%cb0, %cb1) : (memref<128x128xbf16, #ttcore.memory_space<l1>>, memref<4x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) -> ()
    }
    memref.dealloc %alloc_0 : memref<1x1x128x128xbf16, #ttcore.shard<256x2>, #ttcore.memory_space<l1>>
    %alloc_1 = memref.alloc() {address = 100352 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%alloc : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #ttcore.memory_space<l1>>)
        outs(%alloc_1 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, %cb1: memref<4x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>):
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.for %arg1 = %c0 to %c4 step %c2 {
        scf.for %arg2 = %c0 to %c4 step %c4 {
          %subview = memref.subview %cb0[%arg1, %arg2] [2, 4] [1, 1] : memref<4x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>> to memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_4 = memref.subview %cb1[%arg1, %arg2] [2, 4] [1, 1] : memref<4x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>> to memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #ttcore.memory_space<l1>>
          %dst = d2m.acquire_dst() : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %subview[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #ttcore.memory_space<l1>>
              affine.store %0, %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
            }
          }
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
              %1 = "d2m.tile_abs"(%0) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: %[[ABS_RESULT:.*]] = "d2m.tile_abs"(%[[DST0_VAL:.*]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: affine.store %[[ABS_RESULT]], %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              affine.store %1, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
              %2 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
              // CHECK: %[[DST_ABS:.*]] = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              %3 = "d2m.tile_sin"(%2) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: %[[SIN_RESULT:.*]] = "d2m.tile_sin"(%[[DST_ABS]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: affine.store %[[SIN_RESULT]], %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              affine.store %3, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
              %4 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
              // CHECK: %[[DST_SIN:.*]] = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              %5 = "d2m.tile_negative"(%4) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: %[[NEG_RESULT:.*]] = "d2m.tile_negative"(%[[DST_SIN]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: affine.store %[[NEG_RESULT]], %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              affine.store %5, %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
              %6 = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
              // CHECK: %[[DST_NEG:.*]] = affine.load %dst[%c0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              %7 = "d2m.tile_exp"(%6) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: %[[EXP_RESULT:.*]] = "d2m.tile_exp"(%[[DST_NEG]]) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
              // CHECK: affine.store %[[EXP_RESULT]], %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              affine.store %7, %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
            }
          }
          affine.for %arg3 = 0 to 2 {
            affine.for %arg4 = 0 to 4 {
              %0 = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<dst>>
              // CHECK: %[[FINAL_VAL:.*]] = affine.load %dst[0, %arg3, %arg4] : memref<1x2x4x!ttcore.tile<32x32, bf16>, #dst>
              // CHECK: affine.store %[[FINAL_VAL]], %subview_4[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #l1>
              affine.store %0, %subview_4[%arg3, %arg4] : memref<2x4x!ttcore.tile<32x32, bf16>, strided<[4, 1], offset: ?>, #ttcore.memory_space<l1>>
            }
          }
        }
      }
    }
    memref.dealloc %alloc : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #ttcore.memory_space<l1>>
    %alloc_2 = memref.alloc() : memref<128x128xbf16>
    %alloc_3 = memref.alloc() {address = 133120 : i64, alignment = 16 : i64} : memref<1x1x128x128xbf16, #ttcore.shard<256x2>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%alloc_1 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #ttcore.memory_space<l1>>)
        outs(%alloc_3 : memref<1x1x128x128xbf16, #ttcore.shard<256x2>, #ttcore.memory_space<l1>>)  {
    ^compute0(%cb0: memref<4x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, %cb1: memref<128x128xbf16, #ttcore.memory_space<l1>>):
      "d2m.tile_untilize_block"(%cb0, %cb1) : (memref<4x4x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<128x128xbf16, #ttcore.memory_space<l1>>) -> ()
    }
    memref.dealloc %alloc_1 : memref<1x1x4x4x!ttcore.tile<32x32, bf16>, #ttcore.shard<8192x2048>, #ttcore.memory_space<l1>>
    d2m.to_layout %alloc_3, %alloc_2 : memref<1x1x128x128xbf16, #ttcore.shard<256x2>, #ttcore.memory_space<l1>> into memref<128x128xbf16> hostInfo = <logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
    memref.dealloc %alloc_3 : memref<1x1x128x128xbf16, #ttcore.shard<256x2>, #ttcore.memory_space<l1>>
    return %alloc_2 : memref<128x128xbf16>
  }
}
