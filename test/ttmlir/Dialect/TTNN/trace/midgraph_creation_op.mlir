// RUN: ttmlir-opt --ttnn-trace-hoist-transform -o %t %s
// RUN: FileCheck %s --input-file=%t

// A creation op (ttnn.full) materialized in the *middle* of the graph, between
// two hoistable compute ops (ttnn.add, ttnn.multiply). This mirrors what the
// SDPA decomposition emits (ttnn.full for the scale, ttnn.constant for the
// causal mask) when const-eval is disabled, so there is no const-eval pass to
// relocate it. Such a creation op is not hoistable, but it only depends on the
// device (no data dependency on surrounding compute). The trace-hoist pass must
// sink it above the trace region and pass it in as a regular trace input,
// instead of failing with "Non-hoistable op found in the middle of hoistable
// ops".
//
// NOTE: this must be exercised at the pass level. A full-pipeline test using a
// leading-eligible creation op (e.g. ttir.ones with no real operands) gets
// scheduled to the top of the block before trace-hoist runs, so it never
// reaches the mid-graph guard and would pass even without the fix.

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  ttcore.device_module {
    builtin.module attributes {ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

      // The mid-graph ttnn.full is sunk out of the trace and passed in as an arg.
      // CHECK-LABEL: func.func private @trace_0_midgraph
      // CHECK: "ttnn.add"
      // CHECK: "ttnn.multiply"
      // CHECK-NOT: "ttnn.full"

      // CHECK-LABEL: func.func @midgraph(
      func.func @midgraph(%arg0: tensor<1x12x32x64xbf16, #ttnn_layout>, %arg1: tensor<1x12x32x64xbf16, #ttnn_layout>) -> tensor<1x12x32x64xbf16, #ttnn_layout> attributes {tt.function_type = "forward_device"} {
        // The creation op is hoisted above the trace op and host-staged, then fed
        // in as a trace input alongside the two real inputs.
        // CHECK: "ttnn.get_device"
        // CHECK: %[[FULL:.+]] = "ttnn.full"
        // CHECK: %[[FULL_HOST:.+]] = "ttnn.to_layout"(%[[FULL]])
        // CHECK: "ttnn.capture_or_execute_trace"({{.*}}%[[FULL_HOST]])
        %dev = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %add = "ttnn.add"(%arg0, %arg1) : (tensor<1x12x32x64xbf16, #ttnn_layout>, tensor<1x12x32x64xbf16, #ttnn_layout>) -> tensor<1x12x32x64xbf16, #ttnn_layout>
        %full = "ttnn.full"(%dev) <{fill_value = 1.250000e-01 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x12x32x64>}> : (!ttnn.device) -> tensor<1x12x32x64xbf16, #ttnn_layout>
        %mul = "ttnn.multiply"(%add, %full) : (tensor<1x12x32x64xbf16, #ttnn_layout>, tensor<1x12x32x64xbf16, #ttnn_layout>) -> tensor<1x12x32x64xbf16, #ttnn_layout>
        return %mul : tensor<1x12x32x64xbf16, #ttnn_layout>
      }
    }
  }
}
