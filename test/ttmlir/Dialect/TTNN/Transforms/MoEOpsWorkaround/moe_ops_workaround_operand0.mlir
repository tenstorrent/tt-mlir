// RUN: ttmlir-opt --ttnn-moe-ops-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test the operand 0 (input_tensor) case where both reshape AND
// to_memory_config are needed, because source is DRAM interleaved (3-D) and
// destination is L1 interleaved (2-D).

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

// dispatch_metadata result 0: 3-D DRAM interleaved
#disp0 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x2880xbf16, #dram>, <interleaved>>
// moe_gpt operand 0: 2-D L1 interleaved
#moe0 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x2880xbf16, #l1>, <interleaved>>

// Other dispatch/moe layouts we don't transform (kept minimal).
#disp1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x4xui16, #l1>, <height_sharded>>
#moe1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x4xui16, #l1>, <height_sharded>>
#disp2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x4xbf16, #l1>, <height_sharded>>
#moe2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x4xbf16, #l1>, <height_sharded>>
#mapping = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xui16, #l1>, <interleaved>>
#w0w1 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5), <1x1>, memref<12x1x4x4x91x4x!ttcore.tile<32x32, bfp_bf4>, #dram>, <height_sharded>>
#w2 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5), <1x1>, memref<12x1x4x2x91x4x!ttcore.tile<32x32, bfp_bf4>, #dram>, <height_sharded>>

// moe_gpt output layouts.
#tc_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xui32, #l1>, <interleaved>>
#ar_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1548xui32, #l1>, <interleaved>>
#ti_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x516xui32, #l1>, <interleaved>>
#tz_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<72x2x1x90x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // CHECK-LABEL: func.func @test_operand0_reshape_and_mem_cfg
  // CHECK: %[[DISPATCHED:.*]], %[[INDICES:.*]], %[[SCORES:.*]] = "ttnn.all_to_all_dispatch_metadata"
  // CHECK-NOT: "ttnn.pad"
  // CHECK-NOT: "ttnn.slice_static"
  // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[DISPATCHED]]) <{shape = [128 : i32, 2880 : i32]}>
  // CHECK: %[[TMC:.*]] = "ttnn.to_memory_config"(%[[RESHAPE]]) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>
  // CHECK: "ttnn.moe_gpt"(%[[TMC]], %{{.*}}, %{{.*}}
  func.func @test_operand0_reshape_and_mem_cfg(
      %input: tensor<1x1x32x2880xbf16, #disp0>,
      %indices: tensor<1x1x32x4xui16, #disp1>,
      %scores: tensor<1x1x32x4xbf16, #disp2>,
      %mapping: tensor<32x128xui16, #mapping>,
      %moe_indices: tensor<128x4xui16, #moe1>,
      %moe_scores: tensor<128x4xbf16, #moe2>,
      %w0w1: tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>,
      %w2: tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>
  ) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>) {
    %dispatched, %disp_indices, %disp_scores = "ttnn.all_to_all_dispatch_metadata"(%input, %indices, %scores, %mapping) <{cluster_axis = 0 : i64, num_devices = 4 : i64}> : (tensor<1x1x32x2880xbf16, #disp0>, tensor<1x1x32x4xui16, #disp1>, tensor<1x1x32x4xbf16, #disp2>, tensor<32x128xui16, #mapping>) -> (tensor<1x128x2880xbf16, #disp0>, tensor<1x128x4xui16, #disp1>, tensor<1x128x4xbf16, #disp2>)

    // Long chain that should collapse. Source (dispatched) is 3-D DRAM
    // interleaved; destination (moe_gpt operand 0) is 2-D L1 interleaved.
    // Need reshape (shape) + to_memory_config (DRAM -> L1).
    %r1 = "ttnn.reshape"(%dispatched) <{shape = [128 : i32, 2880 : i32]}> : (tensor<1x128x2880xbf16, #disp0>) -> tensor<128x2880xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x2880xbf16, #dram>, <interleaved>>>
    %r2 = "ttnn.reshape"(%r1) <{shape = [128 : i32, 2880 : i32]}> : (tensor<128x2880xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x2880xbf16, #dram>, <interleaved>>>) -> tensor<128x2880xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x2880xbf16, #dram>, <interleaved>>>
    %tmc = "ttnn.to_memory_config"(%r2) <{memory_config = #ttnn.memory_config<#l1, <interleaved>>}> : (tensor<128x2880xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x2880xbf16, #dram>, <interleaved>>>) -> tensor<128x2880xbf16, #moe0>

    %tc, %ar, %ti, %tz1, %tz2 = "ttnn.moe_gpt"(%tmc, %moe_indices, %moe_scores, %mapping, %w0w1, %w2) <{cluster_axis = 0 : ui32, hidden_size = 2880 : ui32, output_height_shard_dim = 4 : ui32, output_width_shard_dim = 3 : ui32}> : (tensor<128x2880xbf16, #moe0>, tensor<128x4xui16, #moe1>, tensor<128x4xbf16, #moe2>, tensor<32x128xui16, #mapping>, tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>, tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>)
    return %tc, %ar, %ti, %tz1, %tz2 : tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>
  }
}
