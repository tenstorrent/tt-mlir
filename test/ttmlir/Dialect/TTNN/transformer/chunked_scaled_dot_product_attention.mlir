// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="composite-resolution=force-promote" %s | FileCheck %s
// CHECK-DAG: #[[PAGE_TABLE_RM_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x4xsi32, #dram>
// CHECK-DAG: #[[CHUNK_START_RM_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x1xsi32, #dram>

// Resolve a ttcore.composite "chunked_scaled_dot_product_attention" through the
// TTNN backend pipeline. The composite is promoted to the typed
// ttnn.chunked_scaled_dot_product_attention op by TTNNResolveComposites, and the
// page_table / chunk_start_idx operands are forced to ROW_MAJOR at layout time
// (a permanent tt-metal kernel ABI, see issue #8842). Note the function args
// carry NO ttcore.argument_type marking: the ROW_MAJOR coercion fires because it
// is keyed off the (composite) op's operands, not off argument marking.
module attributes {} {
  // query: [num_users, num_heads, chunk_len, head_size]; key/value: paged cache
  // [num_blocks, num_kv_heads, block_size, head_size].
  func.func @chunked_sdpa(%arg0: tensor<1x12x64x64xf32>, %arg1: tensor<128x12x32x64xf32>, %arg2: tensor<128x12x32x64xf32>, %arg3: tensor<1x4xi32>, %arg4: tensor<1xi32>) -> tensor<1x12x64x64xf32> {
    // CHECK-LABEL: @chunked_sdpa
    // page_table and chunk_start_idx are forced to ROW_MAJOR for the ttnn kernel.
    // CHECK-DAG: %[[PAGE_TABLE:.*]] = "ttnn.to_layout"(%arg3){{.*}} -> tensor<1x4xsi32, #[[PAGE_TABLE_RM_LAYOUT]]>
    // CHECK-DAG: %[[CHUNK_START:.*]] = "ttnn.to_layout"(%arg4){{.*}} -> tensor<1xsi32, #[[CHUNK_START_RM_LAYOUT]]>
    // CHECK: "ttnn.chunked_scaled_dot_product_attention"
    // CHECK-SAME: %[[PAGE_TABLE]], %[[CHUNK_START]]
    // CHECK-SAME: <{scale = 1.250000e-01 : f32}>
    // CHECK-NOT: "ttcore.composite"
    %0 = "ttcore.composite"(%arg0, %arg1, %arg2, %arg3, %arg4) <{composite_name = "chunked_scaled_dot_product_attention", decomposition = @chunked_scaled_dot_product_attention_decomp, composite_attributes = {scale = 1.250000e-01 : f32}}> : (tensor<1x12x64x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1xi32>) -> tensor<1x12x64x64xf32>
    return %0 : tensor<1x12x64x64xf32>
  }

  // Lean (verify-only) fallback decomposition: identity over the query. Real
  // numeric correctness comes from the promoted typed op; this body is only
  // inlined in a no-OpModel build and just needs to type-check and verify.
  func.func private @chunked_scaled_dot_product_attention_decomp(%query: tensor<1x12x64x64xf32>, %key: tensor<128x12x32x64xf32>, %value: tensor<128x12x32x64xf32>, %page_table: tensor<1x4xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x12x64x64xf32> {
    return %query : tensor<1x12x64x64xf32>
  }
}
