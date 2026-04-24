// RUN: not ttmlir-opt --ttcore-register-device --canonicalize %s 2>&1 | FileCheck %s
#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
#remapping = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0 * 16 + d1, d2 * 2 + d3, d4, d5, d6, d7)>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12}], [0], [1 : i32], [ 0x0x0x0]>
module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @l1_view_layout_output_nd_virtual_grid() {
    %src = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (((d2 + d3 + d1) floordiv 8 + d0 * 2) mod 6, (d2 + d3 + d1) mod 8, d4, d5, d6, d7)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, ((d1 + d0 * 8) floordiv 16) mod 3, (d1 + d0 * 8) mod 16, 0, 0)>} : memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096x4096x4096, 1>, #l1>
    %backing = memref.alloc() {d2m.virtualGridForwardMapping = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (((d2 + d3 + d1) floordiv 8 + d0 * 2) mod 6, (d2 + d3 + d1) mod 8, d4, d5, d6, d7)>, d2m.virtualGridInverseMapping = affine_map<(d0, d1) -> (0, ((d1 + d0 * 8) floordiv 16) mod 3, (d1 + d0 * 8) mod 16, 0, 0)>} : memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096x4096x4096, 1>, #l1>

    // The output view reshapes the backing L1 shard but does not change
    // physical placement. Its apparent 8D grid has volume 96, which exceeds
    // the 8x8 device capacity and cannot be legally collapsed to 2D.
    %view = d2m.view_layout %backing remapping = #remapping : memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096x4096x4096, 1>, #l1> -> memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<8>, #l1>

    // CHECK: error: 'd2m.generic' op output grid shape does not match implied virtual grid shape from physical grid and inverse mapping
    // CHECK: note: see current operation:
    // CHECK-NEXT: "d2m.generic"
    d2m.generic {block_factors = [1, 1, 1, 1], grid = #ttcore.grid<3x16x1x1, virt_to_physical_map = (d0, d1, d2, d3) -> (0, ((d2 + d3 + d1) floordiv 8 + d0 * 2) mod 6, (d2 + d3 + d1) mod 8), physical_to_virt_map = (d0, d1) -> (0, ((d1 + d0 * 8) floordiv 16) mod 3, (d1 + d0 * 8) mod 16, 0, 0)>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel, #parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%src : memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096x4096x4096, 1>, #l1>)
        outs(%view : memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<8>, #l1>)
     {
    ^unified0:
      %block0 = d2m.block_index(0) : index
      %block1 = d2m.block_index(1) : index
      %block2 = d2m.block_index(2) : index
      %block3 = d2m.block_index(3) : index
      %local = memref.alloc() {alignment = 64 : i64} : memref<1x2x1x1x!ttcore.tile<32x32, f32>>
      %loaded = d2m.remote_load %local %src[%block0, %block1, %block2, %block3] : memref<1x2x1x1x!ttcore.tile<32x32, f32>>, memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096x4096x4096, 1>, #l1> -> memref<1x2x1x1x!ttcore.tile<32x32, f32>, #l1>
      %stored = d2m.remote_store %view[%block0, %block1, %block2, %block3] %local : memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<8>, #l1>, memref<1x2x1x1x!ttcore.tile<32x32, f32>> -> memref<3x16x1x1x1x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<8>, #l1>
    }
    return
  }
}
