// RUN: ttmlir-opt --ttnn-decomposition %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

      // dit_rms_norm_unary_fused(input, weight) with silu -> rms_norm + silu.
      // CHECK-LABEL: func.func @decompose_silu_weight
      func.func @decompose_silu_weight(%arg0: tensor<32x512xf32, #ttnn_layout>, %arg1: tensor<512xf32, #ttnn_layout1>) -> tensor<32x512xf32, #ttnn_layout> {
        // CHECK-NOT: "ttnn.dit_rms_norm_unary_fused"
        // CHECK: %[[N:.*]] = "ttnn.rms_norm"(%arg0, %arg1)
        // CHECK: "ttnn.silu"(%[[N]])
        %0 = "ttnn.dit_rms_norm_unary_fused"(%arg0, %arg1) <{activation = "silu", compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<32x512xf32, #ttnn_layout>, tensor<512xf32, #ttnn_layout1>) -> tensor<32x512xf32, #ttnn_layout>
        return %0 : tensor<32x512xf32, #ttnn_layout>
      }

      // With a residual input, the pre-add is materialized before rms_norm.
      // CHECK-LABEL: func.func @decompose_residual
      func.func @decompose_residual(%arg0: tensor<32x512xf32, #ttnn_layout>, %arg1: tensor<512xf32, #ttnn_layout1>, %arg2: tensor<32x512xf32, #ttnn_layout>) -> tensor<32x512xf32, #ttnn_layout> {
        // CHECK-NOT: "ttnn.dit_rms_norm_unary_fused"
        // CHECK: %[[X:.*]] = "ttnn.add"(%arg0, %arg2)
        // CHECK: %[[N:.*]] = "ttnn.rms_norm"(%[[X]], %arg1)
        // CHECK: "ttnn.gelu"(%[[N]])
        %0 = "ttnn.dit_rms_norm_unary_fused"(%arg0, %arg1, %arg2) <{activation = "gelu", compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, epsilon = 9.99999974E-6 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 1>}> : (tensor<32x512xf32, #ttnn_layout>, tensor<512xf32, #ttnn_layout1>, tensor<32x512xf32, #ttnn_layout>) -> tensor<32x512xf32, #ttnn_layout>
        return %0 : tensor<32x512xf32, #ttnn_layout>
      }
    }
  }
}
