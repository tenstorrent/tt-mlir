// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttnn-trace-hoist-transform -o %t %s
// RUN: FileCheck %s --input-file=%t

// ttnn.adamw reads its lr and bias-correction operands back to host, because
// ttml takes them as floats. A trace is captured once and replayed, so a
// captured readback would freeze beta^t at the capturing step's value and every
// later step would silently use it. The optimizer step must therefore stay
// outside the trace region: the forward compute is traced, the adamw op is not.

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#scalar_layout = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xf32, #dram>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

      // Only the forward compute is hoisted.
      // CHECK-LABEL: func.func private @trace_0_fwd_then_step
      // CHECK: "ttnn.add"
      // CHECK-NOT: "ttnn.adamw"

      // CHECK-LABEL: func.func @fwd_then_step(
      func.func @fwd_then_step(%param: tensor<1x1x64x64xf32, #ttnn_layout>, %grad: tensor<1x1x64x64xf32, #ttnn_layout>,
                               %exp_avg: tensor<1x1x64x64xf32, #ttnn_layout>, %exp_avg_sq: tensor<1x1x64x64xf32, #ttnn_layout>,
                               %lr: tensor<1xf32, #scalar_layout>, %beta1_pow: tensor<1xf32, #scalar_layout>,
                               %beta2_pow: tensor<1xf32, #scalar_layout>) -> tensor<1x1x64x64xf32, #ttnn_layout> attributes {tt.function_type = "forward_device"} {
        // The trace call covers the add; the adamw op is left behind it, in the
        // outer function, so its readback runs every step.
        // CHECK: "ttnn.capture_or_execute_trace"
        // CHECK: "ttnn.adamw"
        %act = "ttnn.add"(%param, %grad) : (tensor<1x1x64x64xf32, #ttnn_layout>, tensor<1x1x64x64xf32, #ttnn_layout>) -> tensor<1x1x64x64xf32, #ttnn_layout>
        "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{
            beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
            epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
            : (tensor<1x1x64x64xf32, #ttnn_layout>, tensor<1x1x64x64xf32, #ttnn_layout>,
               tensor<1x1x64x64xf32, #ttnn_layout>, tensor<1x1x64x64xf32, #ttnn_layout>,
               tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>) -> ()
        return %act : tensor<1x1x64x64xf32, #ttnn_layout>
      }
    }
  }
}
