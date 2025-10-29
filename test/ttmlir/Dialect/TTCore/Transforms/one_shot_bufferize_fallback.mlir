// RUN: ttmlir-opt --ttcore-one-shot-bufferize --split-input-file -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that ttcore-one-shot-bufferize properly handles fallback cases:
// 1. Tensors without MetalLayoutAttr encoding (use default identity layout).
// 2. Global ops that are already memref types (no conversion needed).
// 3. Unranked tensors (use fallback conversion).
//
// Note: Each test split needs its own #system_desc due to --split-input-file.

// -----
// Test 1: Function argument with ranked tensor without MetalLayoutAttr encoding.
// This should use the default identity layout conversion.

#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

// CHECK-LABEL: func.func @tensor_without_metal_layout
// CHECK-SAME: (%[[ARG0:.*]]: memref<64x128xf32>) -> memref<64x128xf32>
// CHECK-NOT: tensor<
// CHECK: return %[[ARG0]] : memref<64x128xf32>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  func.func @tensor_without_metal_layout(
    %arg0: tensor<64x128xf32>
  ) -> tensor<64x128xf32> {
    return %arg0 : tensor<64x128xf32>
  }
}

// -----
// Test 2: Global op that is already a memref type (no-op).

#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#mem_space = #ttcore.memory_space<l1>

// CHECK-LABEL: ttcore.global @already_memref
// CHECK-SAME: = memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE:.*]]> [0]
// CHECK-NOT: tensor<

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

      ttcore.global @already_memref = memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #mem_space> [0]
    }
  }
}

// -----
// Test 3: Mixed case - function with tensor without MetalLayoutAttr and alloc operations.

#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

// CHECK-LABEL: func.func @mixed_tensors
// CHECK-SAME: (%[[ARG0:.*]]: memref<32x64xf32>) -> memref<32x64xf32>
// CHECK-NOT: tensor<
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32>
// CHECK: linalg.copy ins(%[[ARG0]] : memref<32x64xf32>) outs(%[[ALLOC]] : memref<32x64xf32>)
// CHECK: return %[[ALLOC]] : memref<32x64xf32>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  func.func @mixed_tensors(
    %arg0: tensor<32x64xf32>
  ) -> tensor<32x64xf32> {
    %0 = bufferization.alloc_tensor() : tensor<32x64xf32>
    %1 = linalg.copy ins(%arg0 : tensor<32x64xf32>) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }
}

// -----
// Test 4: Function argument with unranked tensor.
// This tests the fallback path for unranked tensors (lines 23-26 in BufferizationTypeConverter.h).

#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

// CHECK-LABEL: func.func @unranked_tensor
// CHECK-SAME: (%[[ARG0:.*]]: memref<*xf32>) -> memref<*xf32>
// CHECK-NOT: tensor<
// CHECK: return %[[ARG0]] : memref<*xf32>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  func.func @unranked_tensor(
    %arg0: tensor<*xf32>
  ) -> tensor<*xf32> {
    %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<*xf32>
    return %0 : tensor<*xf32>
  }
}

