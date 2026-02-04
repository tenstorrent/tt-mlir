// RUN: ttmlir-opt --ttcore-register-device --d2m-decompose-masking -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test d2m-decompose-masking pass: decompose block_mask into scf.for loops
// with per-tile masking using TileWhereOp for efficient element-wise selection.

#l1 = #ttcore.memory_space<l1>
#shard_2x2 = #ttcore.shard<8192x4096, 1>
#shard_1x1 = #ttcore.shard<4096x4096, 1>
#shard_4x4 = #ttcore.shard<32768x8192, 1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, 0)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]>

  // CHECK-LABEL: func.func @decompose_block_mask_partial_row
  // Test partial row masking: logical shape 50x64 on a 2x2 tile grid
  func.func @decompose_block_mask_partial_row(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                               %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                               %arg2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>,
                                               %arg3: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>) {
    // CHECK-NOT: d2m.block_mask
    // CHECK: d2m.generic
    // CHECK: d2m.write_row_mask_tile
    // CHECK: d2m.write_col_mask_tile
    // CHECK: scf.for
    // CHECK:   scf.for
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map1, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %arg2, %arg3 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>)
        outs(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c50 = arith.constant 50 : index
      %buf0 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %buf1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %buf2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %buf3 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %buf0 %arg0[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_load %buf1 %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.remote_load %buf2 %arg3[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %4 = "d2m.block_mask"(%0, %buf3, %1, %2, %c50, %c64) <{fill_value = #ttcore.oob_val<inf>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index) -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %5 = d2m.remote_store %arg1[%c0, %c0] %4 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    }
    return
  }

  // CHECK-LABEL: func.func @decompose_block_mask_partial_both
  // Test partial masking in both dimensions: logical shape 50x50 on a 2x2 tile grid
  func.func @decompose_block_mask_partial_both(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                                %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                                %arg2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>,
                                                %arg3: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>) {
    // CHECK-NOT: d2m.block_mask
    // CHECK: d2m.generic
    // CHECK: d2m.write_row_mask_tile
    // CHECK: d2m.write_col_mask_tile
    // Interior loop
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     memref.load
    // CHECK:     memref.store
    // Row edge loop
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     d2m.tile_where
    // Col edge loop
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     d2m.tile_where
    // Corner loop
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     d2m.tile_where
    // CHECK:     d2m.tile_where
    // OOB rows fill
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     d2m.tile_fill
    // OOB cols fill
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     d2m.tile_fill
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map1, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %arg2, %arg3 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>)
        outs(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c50 = arith.constant 50 : index
      %buf0 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %buf1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %buf2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %buf3 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %buf0 %arg0[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_load %buf1 %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.remote_load %buf2 %arg3[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %4 = "d2m.block_mask"(%0, %buf3, %1, %2, %c50, %c50) <{fill_value = #ttcore.oob_val<zero>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index) -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %5 = d2m.remote_store %arg1[%c0, %c0] %4 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    }
    return
  }

  // CHECK-LABEL: func.func @decompose_block_mask_neginf_fill
  // Test with neginf fill value (used for max reductions)
  func.func @decompose_block_mask_neginf_fill(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                               %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                               %arg2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>,
                                               %arg3: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>) {
    // CHECK-NOT: d2m.block_mask
    // CHECK: d2m.generic
    // CHECK: scf.for
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map1, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %arg2, %arg3 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>)
        outs(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c50 = arith.constant 50 : index
      %buf0 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %buf1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %buf2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %buf3 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %buf0 %arg0[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_load %buf1 %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.remote_load %buf2 %arg3[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %4 = "d2m.block_mask"(%0, %buf3, %1, %2, %c50, %c50) <{fill_value = #ttcore.oob_val<neginf>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index) -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %5 = d2m.remote_store %arg1[%c0, %c0] %4 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    }
    return
  }

  // CHECK-LABEL: func.func @decompose_block_mask_aligned
  // Test with aligned dimensions: logical shape 64x64 on 2x2 tile grid (no masking needed)
  func.func @decompose_block_mask_aligned(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                           %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                           %arg2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>,
                                           %arg3: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>) {
    // CHECK-NOT: d2m.block_mask
    // CHECK: d2m.generic
    // CHECK: scf.for
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map1, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %arg2, %arg3 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>)
        outs(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>):
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %buf0 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %buf1 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %buf2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %buf3 = memref.alloc() : memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %0 = d2m.remote_load %buf0 %arg0[%c0, %c0] : memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.remote_load %buf1 %arg2[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.remote_load %buf2 %arg3[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %4 = "d2m.block_mask"(%0, %buf3, %1, %2, %c64, %c64) <{fill_value = #ttcore.oob_val<zero>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index) -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %5 = d2m.remote_store %arg1[%c0, %c0] %4 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    }
    return
  }

  // CHECK-LABEL: func.func @decompose_block_mask_single_tile_valid
  // Test with logical shape 32x32 - only one tile is valid, rest are OOB
  func.func @decompose_block_mask_single_tile_valid(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                                     %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                                     %arg2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>,
                                                     %arg3: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>) {
    // CHECK-NOT: d2m.block_mask
    // CHECK: d2m.generic
    // CHECK: scf.for
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map1, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %arg2, %arg3 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>)
        outs(%arg1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>):
      %c32 = arith.constant 32 : index
      %0 = d2m.wait %cb0 : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.wait %cb2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %3 = d2m.reserve %cb3 : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %4 = "d2m.block_mask"(%0, %3, %1, %2, %c32, %c32) <{fill_value = #ttcore.oob_val<zero>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index) -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

  // CHECK-LABEL: func.func @decompose_block_mask_larger_grid
  // Test with 4x4 tile grid and partial logical shape
  func.func @decompose_block_mask_larger_grid(%arg0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1>,
                                               %arg1: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1>,
                                               %arg2: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>,
                                               %arg3: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>) {
    // CHECK-NOT: d2m.block_mask
    // CHECK: d2m.generic
    // CHECK: scf.for
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map1, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0, %arg2, %arg3 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #shard_1x1, #l1>)
        outs(%arg1 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1>) {
    ^unified0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb3: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
      %c100 = arith.constant 100 : index
      %0 = d2m.wait %cb0 : <memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.wait %cb2 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %3 = d2m.reserve %cb3 : <memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
      %4 = "d2m.block_mask"(%0, %3, %1, %2, %c100, %c100) <{fill_value = #ttcore.oob_val<zero>, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1>}> : (memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, index, index) -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }
}
