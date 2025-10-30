// RUN: ttmlir-opt --ttcore-one-shot-bufferize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that ttcore-one-shot-bufferize properly handles d2m.stream_layout operations
// with MetalLayoutAttr encoding, converting them to memref types with layout attrs.
// This is a regression test for the original crash in getBufferType.

// CHECK: #[[MEM_SPACE:.*]] = #ttcore.memory_space<[[MEM_SPACE_NAME:l1]]>
// CHECK: module attributes {ttcore.system_desc = #system_desc} {
// CHECK:   ttcore.device_module {
// CHECK:     builtin.module attributes {ttcore.system_desc = #system_desc} {
// CHECK:       ttcore.device @default_device
// CHECK:       func.func @stream_layout_test(%[[ARG0:.*]]: memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>) -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]> {
// CHECK-NEXT:    %[[ALLOC:.*]] = memref.alloc() : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
// CHECK-NEXT:    %[[STREAM:.*]] = "d2m.stream_layout"(%[[ARG0]], %[[ALLOC]]) : (memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>, memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>) -> memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #[[MEM_SPACE]]>
// CHECK-NEXT:    %[[ALLOC_OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
// CHECK-NEXT:    memref.copy %[[STREAM]], %[[ALLOC_OUT]] : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #[[MEM_SPACE]]> to memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>
// CHECK-NEXT:    return %[[ALLOC_OUT]] : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #[[MEM_SPACE]]>


#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 64x64, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101152, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073177056, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [0 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

      func.func @stream_layout_test(
        %arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      ) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {
        %0 = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
        %stream = "d2m.stream_layout"(%arg0, %0) : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
        return %stream : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      }
    }
  }
}
