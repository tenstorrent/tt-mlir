// RUN: ttmlir-opt --ttnn-moe-ops-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verifies the AllToAllDispatchMetadataOp <- TopKRouterGptOp pattern on
// operand 1 (expert_indices). The chain between the two ops may carry an
// existing slice_static; when it does, the pass preserves the slice's
// begins/ends/step by constructing a fresh slice op in the destination
// buffer/memory layout, dropping the rest of the unary chain.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>

// topk_router_gpt output 0 (expert_indices): 32x8 ui16 L1 interleaved
#topk_idx = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x8xui16, #l1>, <interleaved>>
// topk_router_gpt output 1 (expert_weights): 32x8 bf16 L1 interleaved
#topk_w   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x8xbf16, #l1>, <interleaved>>
// topk_router_gpt input 0 (hidden states): 32x2880 bf16 L1 interleaved
#topk_in  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x2880xbf16, #l1>, <interleaved>>
// topk_router_gpt weights: 2880x128 bf16 DRAM interleaved
#topk_wt  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2880x128xbf16, #dram>, <interleaved>>
// topk_router_gpt bias: 32x128 bf16 L1 interleaved
#topk_b   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xbf16, #l1>, <interleaved>>

// Existing slice's input/output layout: 2D DRAM tile.
#slice_l  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>

// dispatch_metadata inputs:
//   input_tensor: 1x1x32x2880 bf16 L1 interleaved
//   expert_indices: 1x1x32x4 ui16 L1 interleaved  <-- THIS is the operand under test
//   expert_scores:  1x1x32x4 bf16 L1 interleaved
//   expert_mapping: 32x128 ui16 L1 interleaved
#dm_in  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x2880xbf16, #l1>, <interleaved>>
#dm_idx = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x128xui16, #l1>, <interleaved>>
#dm_sc  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x128xbf16, #l1>, <interleaved>>
#dm_map = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xui16, #l1>, <interleaved>>

// dispatch_metadata outputs (not under test; just need valid shapes).
#dm_disp_out = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x2880xbf16, #l1>, <interleaved>>
#dm_idx_out  = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x4xui16, #l1>, <height_sharded>>
#dm_sc_out   = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x4xbf16, #l1>, <height_sharded>>

module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

  // Chain with an intermediate slice_static that cuts 8 -> 4. The pass
  // should drop the unary layout/typecast ops and keep only a new slice
  // (in the destination L1 layout) plus a post-reshape from 32x4 to
  // 1x1x32x4.
  //
  // CHECK-LABEL: func.func @test_dispatch_topk_with_slice
  // CHECK: %[[IDX:.*]], %[[WT:.*]] = "ttnn.topk_router_gpt"
  // Intermediate layout/typecast chain should be gone:
  // CHECK-NOT: "ttnn.to_layout"(%[[IDX]])
  // CHECK-NOT: "ttnn.typecast"(%[[IDX]])
  // CHECK: %[[NEW_SLICE:.*]] = "ttnn.slice_static"(%[[IDX]])
  // CHECK-SAME: begins = [0 : i32, 0 : i32]
  // CHECK-SAME: ends = [32 : i32, 4 : i32]
  // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[NEW_SLICE]])
  // CHECK-SAME: shape = [1 : i32, 1 : i32, 32 : i32, 4 : i32]
  // CHECK: "ttnn.all_to_all_dispatch_metadata"(%{{.*}}, %[[RESHAPE]]
  func.func @test_dispatch_topk_with_slice(
      %hidden: tensor<32x2880xbf16, #topk_in>,
      %weight: tensor<2880x128xbf16, #topk_wt>,
      %bias:   tensor<32x128xbf16, #topk_b>,
      %dm_input: tensor<1x1x32x2880xbf16, #dm_in>,
      %dm_scores: tensor<1x1x32x4xbf16, #dm_sc>,
      %dm_mapping: tensor<32x128xui16, #dm_map>
  ) -> (tensor<1x128x2880xbf16, #dm_disp_out>, tensor<1x128x4xui16, #dm_idx_out>, tensor<1x128x4xbf16, #dm_sc_out>) {
    %expert_indices, %expert_weights = "ttnn.topk_router_gpt"(%hidden, %weight, %bias) <{k = 8 : i32, num_experts = 128 : i32}> : (tensor<32x2880xbf16, #topk_in>, tensor<2880x128xbf16, #topk_wt>, tensor<32x128xbf16, #topk_b>) -> (tensor<32x8xui16, #topk_idx>, tensor<32x8xbf16, #topk_w>)

    // Unary chain with a slice in the middle. Everything in this chain
    // should get dropped by the pass; only the slice's begins/ends/step
    // are reused.
    %a = "ttnn.typecast"(%expert_indices) <{dtype = #ttcore.supportedDataTypes<u16>}> : (tensor<32x8xui16, #topk_idx>) -> tensor<32x8xui16, #slice_l>
    %b = "ttnn.slice_static"(%a) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x8xui16, #slice_l>) -> tensor<32x4xui16, #slice_l>
    %c = "ttnn.reshape"(%b) <{shape = [1 : i32, 1 : i32, 32 : i32, 4 : i32]}> : (tensor<32x4xui16, #slice_l>) -> tensor<1x1x32x4xui16, #dm_idx>

    %dispatched, %indices, %scores = "ttnn.all_to_all_dispatch_metadata"(%dm_input, %c, %dm_scores, %dm_mapping) <{cluster_axis = 0 : i64, num_devices = 4 : i64}> : (tensor<1x1x32x2880xbf16, #dm_in>, tensor<1x1x32x4xui16, #dm_idx>, tensor<1x1x32x4xbf16, #dm_sc>, tensor<32x128xui16, #dm_map>) -> (tensor<1x128x2880xbf16, #dm_disp_out>, tensor<1x128x4xui16, #dm_idx_out>, tensor<1x128x4xbf16, #dm_sc_out>)

    return %dispatched, %indices, %scores : tensor<1x128x2880xbf16, #dm_disp_out>, tensor<1x128x4xui16, #dm_idx_out>, tensor<1x128x4xbf16, #dm_sc_out>
  }

  // Negative test: when an op in the chain has multiple users (here %a is
  // used by both the chain and the function return), the pattern must not
  // fire. The chain must survive untouched.
  //
  // CHECK-LABEL: func.func @test_dispatch_topk_multi_use
  // CHECK: %[[IDX:.*]], %{{.*}} = "ttnn.topk_router_gpt"
  // CHECK: %[[A:.*]] = "ttnn.typecast"(%[[IDX]])
  // CHECK: %[[B:.*]] = "ttnn.slice_static"(%[[A]])
  // CHECK: %[[C:.*]] = "ttnn.reshape"(%[[B]])
  // CHECK: "ttnn.all_to_all_dispatch_metadata"(%{{.*}}, %[[C]]
  // CHECK: return %{{.*}}, %[[A]]
  func.func @test_dispatch_topk_multi_use(
      %hidden: tensor<32x2880xbf16, #topk_in>,
      %weight: tensor<2880x128xbf16, #topk_wt>,
      %bias:   tensor<32x128xbf16, #topk_b>,
      %dm_input: tensor<1x1x32x2880xbf16, #dm_in>,
      %dm_scores: tensor<1x1x32x4xbf16, #dm_sc>,
      %dm_mapping: tensor<32x128xui16, #dm_map>
  ) -> (tensor<1x128x2880xbf16, #dm_disp_out>, tensor<32x8xui16, #slice_l>, tensor<1x128x4xui16, #dm_idx_out>, tensor<1x128x4xbf16, #dm_sc_out>) {
    %expert_indices, %expert_weights = "ttnn.topk_router_gpt"(%hidden, %weight, %bias) <{k = 8 : i32, num_experts = 128 : i32}> : (tensor<32x2880xbf16, #topk_in>, tensor<2880x128xbf16, #topk_wt>, tensor<32x128xbf16, #topk_b>) -> (tensor<32x8xui16, #topk_idx>, tensor<32x8xbf16, #topk_w>)
    // %a is used by the chain AND returned directly, so the pattern must skip.
    %a = "ttnn.typecast"(%expert_indices) <{dtype = #ttcore.supportedDataTypes<u16>}> : (tensor<32x8xui16, #topk_idx>) -> tensor<32x8xui16, #slice_l>
    %b = "ttnn.slice_static"(%a) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x8xui16, #slice_l>) -> tensor<32x4xui16, #slice_l>
    %c = "ttnn.reshape"(%b) <{shape = [1 : i32, 1 : i32, 32 : i32, 4 : i32]}> : (tensor<32x4xui16, #slice_l>) -> tensor<1x1x32x4xui16, #dm_idx>
    %dispatched, %indices, %scores = "ttnn.all_to_all_dispatch_metadata"(%dm_input, %c, %dm_scores, %dm_mapping) <{cluster_axis = 0 : i64, num_devices = 4 : i64}> : (tensor<1x1x32x2880xbf16, #dm_in>, tensor<1x1x32x4xui16, #dm_idx>, tensor<1x1x32x4xbf16, #dm_sc>, tensor<32x128xui16, #dm_map>) -> (tensor<1x128x2880xbf16, #dm_disp_out>, tensor<1x128x4xui16, #dm_idx_out>, tensor<1x128x4xbf16, #dm_sc_out>)
    return %dispatched, %a, %indices, %scores : tensor<1x128x2880xbf16, #dm_disp_out>, tensor<32x8xui16, #slice_l>, tensor<1x128x4xui16, #dm_idx_out>, tensor<1x128x4xbf16, #dm_sc_out>
  }
}
