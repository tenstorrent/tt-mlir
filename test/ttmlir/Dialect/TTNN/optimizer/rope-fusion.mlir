// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer="memory-layout-analysis-enabled=true" %s --mlir-print-local-scope | FileCheck %s

// This test verifies that the RoPE (Rotary Position Embedding) operation
// can be sharded by the optimizer when memory layout analysis is enabled.

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 1024 + d2, d3), <1x1>, memref<1024x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

      // CHECK-LABEL: func.func @main
      func.func @main(%arg0: tensor<1x1024x64xbf16, #ttnn_layout>, %arg1: tensor<1x32x1024x64xbf16, #ttnn_layout1>, %arg2: tensor<1x1024x64xbf16, #ttnn_layout>) -> tensor<1x32x1024x64xbf16, #ttnn_layout1> {
        %0 = "ttnn.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout2>
        %1 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x1024x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout2>
        // CHECK: "ttnn.rotary_embedding"{{.*}}height_sharded
        %2 = "ttnn.rotary_embedding"(%arg1, %0, %1) : (tensor<1x32x1024x64xbf16, #ttnn_layout1>, tensor<1x1x1024x64xbf16, #ttnn_layout2>, tensor<1x1x1024x64xbf16, #ttnn_layout2>) -> tensor<1x32x1024x64xbf16, #ttnn_layout1>
        %3 = "ttnn.add"(%2, %2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x32x1024x64xbf16, #ttnn_layout1>, tensor<1x32x1024x64xbf16, #ttnn_layout1>) -> tensor<1x32x1024x64xbf16, #ttnn_layout1>
        return %3 : tensor<1x32x1024x64xbf16, #ttnn_layout1>
      }
    }
  }
}
