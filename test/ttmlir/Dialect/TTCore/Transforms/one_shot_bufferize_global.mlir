// RUN: ttmlir-opt --ttcore-register-device --ttcore-one-shot-bufferize -o %t %s --split-input-file
// RUN: FileCheck %s --input-file=%t

// Test that ttcore-one-shot-bufferize properly converts ttcore.global operations
// from tensor types with MetalLayoutAttr encoding to memref types with proper
// layout attributes and memory space.

// -----
// Test 1: Global inside device_module

// CHECK: module attributes {ttcore.system_desc = #system_desc}
// CHECK:   ttcore.device_module
// CHECK: builtin.module
// CHECK: ttcore.device @default_device
// CHECK-NEXT: ttcore.global @global_in_device_module = memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE:.*]]> [0]

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

      ttcore.global @global_in_device_module = tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> [0]
    }
  }
}

// -----
// Test 2: Global in top-level module without device_module

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

// CHECK: module attributes {ttcore.system_desc = #system_desc}
// CHECK: ttcore.device @default_device
// CHECK-NEXT: ttcore.global @global_in_toplevel = memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE2:.*]]> [0]

module {
  ttcore.global @global_in_toplevel = tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> [0]
}
