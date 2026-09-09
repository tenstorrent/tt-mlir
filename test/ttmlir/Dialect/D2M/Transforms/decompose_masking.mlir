// RUN: ttmlir-opt --ttcore-register-device --d2m-decompose-masking -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test d2m-decompose-masking pass: decompose top-level d2m.mask ops into
// explicit per-core generics with row/column mask tiles and tile loops.

#l1 = #ttcore.memory_space<l1>
#shard_2x2 = #ttcore.shard<8192x4096, 1>
#shard_3d = #ttcore.shard<4096x4096x4096, 1>
#shard_4x4 = #ttcore.shard<32768x8192, 1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]>

  // CHECK-LABEL: func.func @decompose_mask_partial_row
  // Test partial row masking: logical shape 50x64 on a 2x2 tile grid.
  func.func @decompose_mask_partial_row(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                        %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    // CHECK-NOT: d2m.mask
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<1x1>
    // CHECK: d2m.remote_load
    // CHECK: d2m.write_row_mask_tile
    // CHECK-NOT: d2m.write_col_mask_tile
    // CHECK: scf.for
    // CHECK: d2m.tile_where
    // CHECK: d2m.remote_store
    %0 = d2m.mask %arg0, %arg1 logical_shape = [50, 64] fill_value = <inf> : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> into memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    return
  }

  // CHECK-LABEL: func.func @decompose_mask_partial_both
  // Test partial masking in both dimensions: logical shape 50x50 on a 2x2 tile grid.
  func.func @decompose_mask_partial_both(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                         %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    // CHECK-NOT: d2m.mask
    // CHECK: d2m.generic
    // CHECK: d2m.write_row_mask_tile
    // CHECK: d2m.write_col_mask_tile
    // CHECK: scf.for
    // CHECK: memref.load
    // CHECK: memref.store
    // CHECK: d2m.tile_where
    // CHECK: d2m.tile_fill
    // CHECK: d2m.remote_store
    %0 = d2m.mask %arg0, %arg1 logical_shape = [50, 50] fill_value = <zero> : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> into memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    return
  }

  // CHECK-LABEL: func.func @decompose_mask_neginf_fill
  // Test with neginf fill value (used for max reductions).
  func.func @decompose_mask_neginf_fill(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                        %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    // CHECK-NOT: d2m.mask
    // CHECK: d2m.generic
    // CHECK: arith.constant {{.*}} : f32
    // CHECK: d2m.tile_fill
    // CHECK: d2m.remote_store
    %0 = d2m.mask %arg0, %arg1 logical_shape = [50, 50] fill_value = <neginf> : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> into memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    return
  }

  // CHECK-LABEL: func.func @decompose_mask_aligned
  // The decompose pass handles already-aligned memref masks; tensor canonicalize
  // removes these before bufferization in the normal pipeline.
  func.func @decompose_mask_aligned(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                    %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    // CHECK-NOT: d2m.mask
    // CHECK: d2m.generic
    // CHECK: scf.for
    // CHECK: d2m.remote_store
    %0 = d2m.mask %arg0, %arg1 logical_shape = [64, 64] fill_value = <zero> : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> into memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    return
  }

  // CHECK-LABEL: func.func @decompose_mask_single_tile_valid
  // Test with logical shape 32x32: only one tile is valid, rest are OOB.
  func.func @decompose_mask_single_tile_valid(%arg0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>,
                                              %arg1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>) {
    // CHECK-NOT: d2m.mask
    // CHECK: d2m.generic
    // CHECK: d2m.tile_fill
    // CHECK: d2m.remote_store
    %0 = d2m.mask %arg0, %arg1 logical_shape = [32, 32] fill_value = <zero> : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> into memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1> -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #shard_2x2, #l1>
    return
  }

  // CHECK-LABEL: func.func @decompose_mask_larger_shard
  // Test with a 4x4 local shard and partial logical shape.
  func.func @decompose_mask_larger_shard(%arg0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1>,
                                         %arg1: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1>) {
    // CHECK-NOT: d2m.mask
    // CHECK: d2m.generic
    // CHECK: memref.alloc() : memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    // CHECK: d2m.remote_load
    // CHECK: scf.for
    // CHECK: d2m.remote_store
    %0 = d2m.mask %arg0, %arg1 logical_shape = [100, 100] fill_value = <zero> : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1> into memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1> -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #shard_4x4, #l1>
    return
  }

  // CHECK-LABEL: func.func @decompose_mask_rank3_shard
  // CHECK: grid = #ttcore.grid<2x1x2, virt_to_physical_map =
  // CHECK: d2m.core_index(2) : index
  // CHECK-NOT: memref.load %{{[^[]*}}[%{{[^,]*}}, %{{[^,]*}}] : memref<1x1x1x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: memref.load %{{[^[]*}}[%{{[^,]*}}, %{{[^,]*}}, %{{[^]]*}}] : memref<1x1x1x!ttcore.tile<32x32, f32>, #l1>
  // CHECK-NOT: memref.load %{{[^[]*}}[%{{[^,]*}}, %{{[^,]*}}] : memref<1x1x1x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: memref.store %{{[^,]*}}, %{{[^[]*}}[%{{[^,]*}}, %{{[^,]*}}, %{{[^]]*}}] : memref<1x1x1x!ttcore.tile<32x32, f32>, #l1>
  // CHECK-NOT: memref.store %{{[^,]*}}, %{{[^[]*}}[%{{[^,]*}}, %{{[^,]*}}] : memref<1x1x1x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: return
  func.func @decompose_mask_rank3_shard() {
    %input = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d2 floordiv 2 + d1 + d0) mod 2, d2 mod 2, d3, d4, d5)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, (d1 floordiv 2 + d0) mod 2, 0, d1 mod 2)>} : memref<2x1x2x1x1x1x!ttcore.tile<32x32, f32>, #shard_3d, #l1>
    %output = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d2 floordiv 2 + d1 + d0) mod 2, d2 mod 2, d3, d4, d5)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, (d1 floordiv 2 + d0) mod 2, 0, d1 mod 2)>} : memref<2x1x2x1x1x1x!ttcore.tile<32x32, f32>, #shard_3d, #l1>
    %0 = d2m.mask %input, %output logical_shape = [2, 4, 64] fill_value = <zero> : memref<2x1x2x1x1x1x!ttcore.tile<32x32, f32>, #shard_3d, #l1> into memref<2x1x2x1x1x1x!ttcore.tile<32x32, f32>, #shard_3d, #l1> -> memref<2x1x2x1x1x1x!ttcore.tile<32x32, f32>, #shard_3d, #l1>
    return
  }
}
