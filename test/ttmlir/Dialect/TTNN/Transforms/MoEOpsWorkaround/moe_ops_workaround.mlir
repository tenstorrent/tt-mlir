// RUN: ttmlir-opt --ttnn-moe-ops-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// Layouts shared across tests. Shapes/layouts mirror what the MoE pipeline
// actually produces in the gpt-oss 120b IR after layout decomposition:
//   - dispatch_metadata result 0 (dispatched): [1, M, H], DRAM interleaved,
//     row_major
//   - dispatch_metadata result 1 (indices):    [1, M, K], L1 height_sharded,
//     row_major
//   - dispatch_metadata result 2 (scores):     [1, M, K], L1 height_sharded,
//     row_major
//   - moe_gpt operand 0 (input_tensor):        [M, H], L1 interleaved, row_major
//   - moe_gpt operand 1 (expert_indices):      [M, K], L1 height_sharded,
//     row_major
//   - moe_gpt operand 2 (expert_scores):       [M, K], L1 height_sharded,
//     row_major

// dispatch_metadata result 0: 3-D DRAM interleaved
#disp0 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x2880xbf16, #dram>, <interleaved>>
// moe_gpt operand 0: 2-D L1 interleaved
#moe0 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x2880xbf16, #l1>, <interleaved>>
// Intermediate 3-D DRAM tile layout used by the existing chain.
#tile3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<4x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Intermediate 2-D DRAM tile layout.
#tile2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Intermediate 2-D DRAM row_major layout.
#rm2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x2880xbf16, #dram>, <interleaved>>

// dispatch_metadata result 1: 3-D L1 height_sharded
#disp1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x4xui16, #l1>, <height_sharded>>
// moe_gpt operand 1: 2-D L1 height_sharded
#moe1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x4xui16, #l1>, <height_sharded>>

// dispatch_metadata result 2: 3-D L1 height_sharded
#disp2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<128x4xbf16, #l1>, <height_sharded>>
// moe_gpt operand 2: 2-D L1 height_sharded
#moe2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x4xbf16, #l1>, <height_sharded>>

// Other operand layouts used only to satisfy the MoeGptOp signature.
#mapping = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x128xui16, #l1>, <interleaved>>
#w0w1 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5), <1x1>, memref<12x1x4x4x91x4x!ttcore.tile<32x32, bfp_bf4>, #dram>, <height_sharded>>
#w2 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5), <1x1>, memref<12x1x4x2x91x4x!ttcore.tile<32x32, bfp_bf4>, #dram>, <height_sharded>>

// moe_gpt output layouts (arbitrary; not under test).
#tc_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xui32, #l1>, <interleaved>>
#ar_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1548xui32, #l1>, <interleaved>>
#ti_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x516xui32, #l1>, <interleaved>>
#tz_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<72x2x1x90x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>

module attributes {} {

// Test 1: operand 2 (expert_scores) chain that needs only a reshape.
// Chain: scores (3D) -> reshape -> moe_gpt.operand2 (2D).
// Both source and dest are L1 height_sharded row_major; shape differs.
// Expected: chain collapses to a single ttnn.reshape directly from scores.
//
// CHECK-LABEL: func.func @test_operand2_reshape_only
// CHECK: %[[DISPATCHED:.*]], %[[INDICES:.*]], %[[SCORES:.*]] = "ttnn.all_to_all_dispatch_metadata"
// CHECK-NOT: "ttnn.to_layout"
// CHECK-NOT: "ttnn.slice_static"
// CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[SCORES]]) <{shape = [128 : i32, 4 : i32]}>
// CHECK: "ttnn.moe_gpt"(%{{.*}}, %{{.*}}, %[[RESHAPE]]
func.func @test_operand2_reshape_only(
    %input: tensor<1x1x32x2880xbf16, #disp0>,
    %indices: tensor<1x1x32x4xui16, #disp1>,
    %scores: tensor<1x1x32x4xbf16, #disp2>,
    %mapping: tensor<32x128xui16, #mapping>,
    %moe_input: tensor<128x2880xbf16, #moe0>,
    %moe_indices: tensor<128x4xui16, #moe1>,
    %w0w1: tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>,
    %w2: tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>
) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>) {
  %dispatched, %disp_indices, %disp_scores = "ttnn.all_to_all_dispatch_metadata"(%input, %indices, %scores, %mapping) <{cluster_axis = 0 : i64, num_devices = 4 : i64}> : (tensor<1x1x32x2880xbf16, #disp0>, tensor<1x1x32x4xui16, #disp1>, tensor<1x1x32x4xbf16, #disp2>, tensor<32x128xui16, #mapping>) -> (tensor<1x128x2880xbf16, #disp0>, tensor<1x128x4xui16, #disp1>, tensor<1x128x4xbf16, #disp2>)
  // Long chain we want the pass to collapse. The final result type is moe2
  // (2-D L1 height_sharded row_major bf16); source is L1 height_sharded 3-D
  // so only a reshape is needed.
  %a = "ttnn.reshape"(%disp_scores) <{shape = [128 : i32, 4 : i32]}> : (tensor<1x128x4xbf16, #disp2>) -> tensor<128x4xbf16, #moe2>
  %b = "ttnn.reshape"(%a) <{shape = [128 : i32, 4 : i32]}> : (tensor<128x4xbf16, #moe2>) -> tensor<128x4xbf16, #moe2>
  %c = "ttnn.reshape"(%b) <{shape = [128 : i32, 4 : i32]}> : (tensor<128x4xbf16, #moe2>) -> tensor<128x4xbf16, #moe2>
  %tc, %ar, %ti, %tz1, %tz2 = "ttnn.moe_gpt"(%moe_input, %moe_indices, %c, %mapping, %w0w1, %w2) <{cluster_axis = 0 : ui32, hidden_size = 2880 : ui32, output_height_shard_dim = 4 : ui32, output_width_shard_dim = 3 : ui32}> : (tensor<128x2880xbf16, #moe0>, tensor<128x4xui16, #moe1>, tensor<128x4xbf16, #moe2>, tensor<32x128xui16, #mapping>, tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>, tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>)
  return %tc, %ar, %ti, %tz1, %tz2 : tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>
}

// Test 2: operand 1 (expert_indices) chain that needs only a reshape.
// Same setup as test 1 but on operand 1.
//
// CHECK-LABEL: func.func @test_operand1_reshape_only
// CHECK: %[[DISPATCHED:.*]], %[[INDICES:.*]], %[[SCORES:.*]] = "ttnn.all_to_all_dispatch_metadata"
// CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[INDICES]]) <{shape = [128 : i32, 4 : i32]}>
// CHECK: "ttnn.moe_gpt"(%{{.*}}, %[[RESHAPE]]
func.func @test_operand1_reshape_only(
    %input: tensor<1x1x32x2880xbf16, #disp0>,
    %indices: tensor<1x1x32x4xui16, #disp1>,
    %scores: tensor<1x1x32x4xbf16, #disp2>,
    %mapping: tensor<32x128xui16, #mapping>,
    %moe_input: tensor<128x2880xbf16, #moe0>,
    %moe_scores: tensor<128x4xbf16, #moe2>,
    %w0w1: tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>,
    %w2: tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>
) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>) {
  %dispatched, %disp_indices, %disp_scores = "ttnn.all_to_all_dispatch_metadata"(%input, %indices, %scores, %mapping) <{cluster_axis = 0 : i64, num_devices = 4 : i64}> : (tensor<1x1x32x2880xbf16, #disp0>, tensor<1x1x32x4xui16, #disp1>, tensor<1x1x32x4xbf16, #disp2>, tensor<32x128xui16, #mapping>) -> (tensor<1x128x2880xbf16, #disp0>, tensor<1x128x4xui16, #disp1>, tensor<1x128x4xbf16, #disp2>)
  // Multi-step unary chain that should collapse to a single reshape.
  %a = "ttnn.reshape"(%disp_indices) <{shape = [128 : i32, 4 : i32]}> : (tensor<1x128x4xui16, #disp1>) -> tensor<128x4xui16, #moe1>
  %b = "ttnn.reshape"(%a) <{shape = [128 : i32, 4 : i32]}> : (tensor<128x4xui16, #moe1>) -> tensor<128x4xui16, #moe1>
  %tc, %ar, %ti, %tz1, %tz2 = "ttnn.moe_gpt"(%moe_input, %b, %moe_scores, %mapping, %w0w1, %w2) <{cluster_axis = 0 : ui32, hidden_size = 2880 : ui32, output_height_shard_dim = 4 : ui32, output_width_shard_dim = 3 : ui32}> : (tensor<128x2880xbf16, #moe0>, tensor<128x4xui16, #moe1>, tensor<128x4xbf16, #moe2>, tensor<32x128xui16, #mapping>, tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>, tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>)
  return %tc, %ar, %ti, %tz1, %tz2 : tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>
}

// Negative test: when an op in the chain has more than one user, the
// pattern must not fire. Otherwise we would redirect moe_gpt without being
// able to delete the chain ops (the extra user keeps them alive), which
// would only add ops, not remove them.
//
// The direct moe_gpt operand %a is used both by moe_gpt and by the return
// value, so hasOneUse() on %a fails and the pattern is skipped.
//
// CHECK-LABEL: func.func @test_operand2_multi_use_intermediate
// CHECK: %[[DISPATCHED:.*]], %[[INDICES:.*]], %[[SCORES:.*]] = "ttnn.all_to_all_dispatch_metadata"
// CHECK: %[[A:.*]] = "ttnn.reshape"(%[[SCORES]])
// Pattern must not have inserted a new reshape/to_memory_config.
// CHECK-NOT: "ttnn.reshape"(%[[A]])
// CHECK-NOT: "ttnn.to_memory_config"
// CHECK: "ttnn.moe_gpt"(%{{.*}}, %{{.*}}, %[[A]]
// CHECK: return %{{.*}}, %[[A]]
func.func @test_operand2_multi_use_intermediate(
    %input: tensor<1x1x32x2880xbf16, #disp0>,
    %indices: tensor<1x1x32x4xui16, #disp1>,
    %scores: tensor<1x1x32x4xbf16, #disp2>,
    %mapping: tensor<32x128xui16, #mapping>,
    %moe_input: tensor<128x2880xbf16, #moe0>,
    %moe_indices: tensor<128x4xui16, #moe1>,
    %w0w1: tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>,
    %w2: tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>
) -> (tensor<1x4xui32, #tc_out>, tensor<128x4xbf16, #moe2>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>) {
  %dispatched, %disp_indices, %disp_scores = "ttnn.all_to_all_dispatch_metadata"(%input, %indices, %scores, %mapping) <{cluster_axis = 0 : i64, num_devices = 4 : i64}> : (tensor<1x1x32x2880xbf16, #disp0>, tensor<1x1x32x4xui16, #disp1>, tensor<1x1x32x4xbf16, #disp2>, tensor<32x128xui16, #mapping>) -> (tensor<1x128x2880xbf16, #disp0>, tensor<1x128x4xui16, #disp1>, tensor<1x128x4xbf16, #disp2>)
  // %a is used by BOTH moe_gpt (operand 2) AND the function return. The
  // hasOneUse() check must reject the pattern since bypassing moe_gpt's
  // use of %a would leave %a alive (the return still uses it), i.e. we'd
  // be adding ops instead of removing them.
  %a = "ttnn.reshape"(%disp_scores) <{shape = [128 : i32, 4 : i32]}> : (tensor<1x128x4xbf16, #disp2>) -> tensor<128x4xbf16, #moe2>
  %tc, %ar, %ti, %tz1, %tz2 = "ttnn.moe_gpt"(%moe_input, %moe_indices, %a, %mapping, %w0w1, %w2) <{cluster_axis = 0 : ui32, hidden_size = 2880 : ui32, output_height_shard_dim = 4 : ui32, output_width_shard_dim = 3 : ui32}> : (tensor<128x2880xbf16, #moe0>, tensor<128x4xui16, #moe1>, tensor<128x4xbf16, #moe2>, tensor<32x128xui16, #mapping>, tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>, tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>)
  return %tc, %a, %ar, %ti, %tz1, %tz2 : tensor<1x4xui32, #tc_out>, tensor<128x4xbf16, #moe2>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>
}

// Test 3: chain with no intermediate ops should not re-match (idempotent
// after the pattern has already fired).
//
// CHECK-LABEL: func.func @test_operand2_already_minimal
// CHECK: %[[DISPATCHED:.*]], %[[INDICES:.*]], %[[SCORES:.*]] = "ttnn.all_to_all_dispatch_metadata"
// CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[SCORES]])
// CHECK-NOT: "ttnn.reshape"(%[[RESHAPE]])
// CHECK: "ttnn.moe_gpt"(%{{.*}}, %{{.*}}, %[[RESHAPE]]
func.func @test_operand2_already_minimal(
    %input: tensor<1x1x32x2880xbf16, #disp0>,
    %indices: tensor<1x1x32x4xui16, #disp1>,
    %scores: tensor<1x1x32x4xbf16, #disp2>,
    %mapping: tensor<32x128xui16, #mapping>,
    %moe_input: tensor<128x2880xbf16, #moe0>,
    %moe_indices: tensor<128x4xui16, #moe1>,
    %w0w1: tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>,
    %w2: tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>
) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>) {
  %dispatched, %disp_indices, %disp_scores = "ttnn.all_to_all_dispatch_metadata"(%input, %indices, %scores, %mapping) <{cluster_axis = 0 : i64, num_devices = 4 : i64}> : (tensor<1x1x32x2880xbf16, #disp0>, tensor<1x1x32x4xui16, #disp1>, tensor<1x1x32x4xbf16, #disp2>, tensor<32x128xui16, #mapping>) -> (tensor<1x128x2880xbf16, #disp0>, tensor<1x128x4xui16, #disp1>, tensor<1x128x4xbf16, #disp2>)
  // Already minimal: single reshape between dispatch and moe_gpt.
  %r = "ttnn.reshape"(%disp_scores) <{shape = [128 : i32, 4 : i32]}> : (tensor<1x128x4xbf16, #disp2>) -> tensor<128x4xbf16, #moe2>
  %tc, %ar, %ti, %tz1, %tz2 = "ttnn.moe_gpt"(%moe_input, %moe_indices, %r, %mapping, %w0w1, %w2) <{cluster_axis = 0 : ui32, hidden_size = 2880 : ui32, output_height_shard_dim = 4 : ui32, output_width_shard_dim = 3 : ui32}> : (tensor<128x2880xbf16, #moe0>, tensor<128x4xui16, #moe1>, tensor<128x4xbf16, #moe2>, tensor<32x128xui16, #mapping>, tensor<12x1x4x4x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w0w1>, tensor<12x1x4x2x2912x128x!ttcore.tile<32x32, bfp_bf4>, #w2>) -> (tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>)
  return %tc, %ar, %ti, %tz1, %tz2 : tensor<1x4xui32, #tc_out>, tensor<1x1548xui32, #ar_out>, tensor<4x516xui32, #ti_out>, tensor<72x2x32x2880xbf16, #tz_out>, tensor<72x2x32x2880xbf16, #tz_out>
}

}
