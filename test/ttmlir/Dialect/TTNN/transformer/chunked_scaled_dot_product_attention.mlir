// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  // query: [num_users, num_heads, chunk_len, head_size]; key/value: paged cache
  // [num_blocks, num_kv_heads, block_size, head_size].
  func.func @chunked_sdpa(%arg0: tensor<1x12x64x64xf32>, %arg1: tensor<128x12x32x64xf32>, %arg2: tensor<128x12x32x64xf32>, %arg3: tensor<1x4xi32>, %arg4: tensor<1xi32>) -> tensor<1x12x64x64xf32> {
    // CHECK-LABEL: @chunked_sdpa
    // page_table and chunk_start_idx are forced to ROW_MAJOR for the ttnn kernel.
    // CHECK-DAG: "ttnn.to_layout"(%arg3) <{layout = #ttnn.layout<row_major>}>
    // CHECK-DAG: "ttnn.to_layout"(%arg4) <{layout = #ttnn.layout<row_major>}>
    // CHECK: "ttnn.chunked_scaled_dot_product_attention"
    // CHECK-SAME: <{scale = 1.250000e-01 : f32}>
    %0 = ttir.empty() : tensor<1x12x64x64xf32>
    %1 = "ttir.chunked_scaled_dot_product_attention"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1xi32>, tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32>
    return %1 : tensor<1x12x64x64xf32>
  }
}
