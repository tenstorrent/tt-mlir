// RUN: ttmlir-opt --ttcore-one-shot-bufferize --split-input-file -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that ttcore-one-shot-bufferize properly converts function arguments
// with MetalLayoutAttr encoding to memrefs with appropriate layout attrs
// (shard, interleaved), handling various module nesting structures and
// memory spaces (L1, DRAM).


// -----
// Test 1: Standard nesting (module → device_module → builtin.module)

// CHECK: #[[MEM_SPACE:.*]] = #ttcore.memory_space<[[MEM_SPACE_NAME:l1]]>
// CHECK: module attributes {ttcore.system_desc = #system_desc} {
// CHECK:   ttcore.device_module {
// CHECK:     builtin.module attributes {ttcore.system_desc = #system_desc} {
// CHECK:       ttcore.device @default_device
// CHECK:       func.func @nested_device_module(%[[ARG0:.*]]: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>, %[[ARG1:.*]]: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>) -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]> {
// CHECK-NEXT:    return %[[ARG0]] : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
// CHECK-NEXT:  }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

      func.func @nested_device_module(
        %arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>,
        %arg1: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      ) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {
        return %arg0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      }
    }
  }
}

// -----
// Test 2: Simple module (no device_module nesting)

// CHECK: module attributes {ttcore.system_desc = #system_desc} {
// CHECK:   ttcore.device @default_device
// CHECK:   func.func @simple_module(%[[ARG0:.*]]: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>) -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]> {
// CHECK-NEXT:    return %[[ARG0]] : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
// CHECK-NEXT:  }
// CHECK-NEXT: }

#layout1 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#system_desc1 = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc1} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  func.func @simple_module(
    %arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>
  ) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1> {
    return %arg0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>
  }
}

// -----
// Test 3: Interleaved memory layout

// CHECK: #[[MEM_SPACE_DRAM:.*]] = #ttcore.memory_space<[[MEM_SPACE_NAME_DRAM:dram]]>
// CHECK: module attributes {ttcore.system_desc = #system_desc} {
// CHECK:   ttcore.device @default_device
// CHECK:   func.func @interleaved_layout(%[[ARG0:.*]]: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #[[MEM_SPACE_DRAM]]>) -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #[[MEM_SPACE_DRAM]]> {
// CHECK-NEXT:    return %[[ARG0]] : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.interleaved<8192x4096>, #[[MEM_SPACE_DRAM]]>
// CHECK-NEXT:  }
// CHECK-NEXT: }

#layout_interleaved = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram, interleaved>
#system_desc2 = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc2} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  func.func @interleaved_layout(
    %arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_interleaved>
  ) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_interleaved> {
    return %arg0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_interleaved>
  }
}
