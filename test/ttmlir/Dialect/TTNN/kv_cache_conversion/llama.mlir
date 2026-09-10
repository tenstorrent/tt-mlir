// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s
// Llama 3.1 8B decode — GQA: bf16 caches → bfp_bf8; paged_update_cache takes the
// bfp_bf8 cache natively (no typecast on the update input); sdpa_decode is a stop op
// that consumes bfp_bf8 K/V natively alongside the bf16 query.

#dram  = #ttnn.buffer_type<dram>
// K/V cache: 2 pages x 1 kv_head x 32 ctx x 32 head_dim
#cache = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Update / query tensor: 1 x 2 batch x 1 token x 32 head_dim
#upd   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2 + d1 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ptab  = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>

// CHECK-LABEL: func.func @llama
// Cache args converted to bfp_bf8; query and update tensors stay bf16.
// CHECK-SAME: %arg0: tensor<2x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg1: tensor<2x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
module attributes {} {
  func.func @llama(
      %k_cache: tensor<2x1x32x32xbf16, #cache> {ttcore.kv_cache},
      %v_cache: tensor<2x1x32x32xbf16, #cache> {ttcore.kv_cache},
      %query:   tensor<1x2x1x32xbf16, #upd>,
      %new_k:   tensor<1x2x1x32xbf16, #upd>,
      %new_v:   tensor<1x2x1x32xbf16, #upd>,
      %ptab:    tensor<2xsi32, #ptab>
  ) -> tensor<1x2x1x32xbf16, #upd> attributes {tt.function_type = "forward_device"} {
    // paged_update_cache is excluded from typecast insertion; the bfp_bf8 cache is
    // written by the bf16 update tensor without an intermediate cast.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.paged_update_cache"(%arg0, %arg3, %arg5)
    "ttnn.paged_update_cache"(%k_cache, %new_k, %ptab) <{share_cache = false}> : (
        tensor<2x1x32x32xbf16, #cache>,
        tensor<1x2x1x32xbf16, #upd>,
        tensor<2xsi32, #ptab>
    ) -> ()
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.paged_update_cache"(%arg1, %arg4, %arg5)
    "ttnn.paged_update_cache"(%v_cache, %new_v, %ptab) <{share_cache = false}> : (
        tensor<2x1x32x32xbf16, #cache>,
        tensor<1x2x1x32xbf16, #upd>,
        tensor<2xsi32, #ptab>
    ) -> ()
    // sdpa_decode is a stop op: K/V are bfp_bf8, query stays bf16, no typecast.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.scaled_dot_product_attention_decode"(%arg2, %arg0, %arg1)
    %0 = "ttnn.scaled_dot_product_attention_decode"(%query, %k_cache, %v_cache) <{
        is_causal = false,
        operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
        scale = 0.125 : f32
    }> : (
        tensor<1x2x1x32xbf16, #upd>,
        tensor<2x1x32x32xbf16, #cache>,
        tensor<2x1x32x32xbf16, #cache>
    ) -> tensor<1x2x1x32xbf16, #upd>
    return %0 : tensor<1x2x1x32xbf16, #upd>
  }
}
