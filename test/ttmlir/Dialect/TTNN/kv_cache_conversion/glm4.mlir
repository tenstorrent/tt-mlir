// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s
// GLM-4 7B TP decode — explicit attention matmul without fused SDPA:
// repeat_interleave and permute are transparent (bfp_bf8 propagates through);
// the attention matmul is a stop op because Q is bf16 (no typecast inserted).

#dram    = #ttnn.buffer_type<dram>
// K cache: 2 pages x 1 kv_head x 32 ctx x 32 head_dim
#cache   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// After repeat_interleave(dim=1, repeats=2): 2 pages x 2 heads x 32 ctx x 32 head_dim
#rep     = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Update tensor: 1 x 2 batch x 1 token x 32 head_dim
#upd     = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2 + d1 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Query: 2 batch x 2 q_heads x 1 token x 32 head_dim
#query   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2 + d1 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ptab    = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>

// CHECK-LABEL: func.func @glm4
// Cache converted to bfp_bf8; query stays bf16.
// CHECK-SAME: %arg0: tensor<2x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
module attributes {} {
  func.func @glm4(
      %k_cache: tensor<2x1x32x32xbf16, #cache> {ttcore.kv_cache},
      %query:   tensor<2x2x1x32xbf16, #query>,
      %new_k:   tensor<1x2x1x32xbf16, #upd>,
      %ptab:    tensor<2xsi32, #ptab>
  ) -> tensor<2x2x1x32xbf16, #query> attributes {tt.function_type = "forward_device"} {
    // Write: paged_update_cache, no typecast on update input.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.paged_update_cache"(%arg0, %arg2, %arg3)
    "ttnn.paged_update_cache"(%k_cache, %new_k, %ptab) <{share_cache = false}> : (
        tensor<2x1x32x32xbf16, #cache>,
        tensor<1x2x1x32xbf16, #upd>,
        tensor<2xsi32, #ptab>
    ) -> ()

    // repeat_interleave is transparent — bfp_bf8 propagates to the result.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[REP:.*]] = "ttnn.repeat_interleave"(%arg0)
    // CHECK-SAME: -> tensor<2x2x32x32x!ttcore.tile<32x32, bfp_bf8>,
    %rep = "ttnn.repeat_interleave"(%k_cache) <{dim = 1 : si32, repeats = 2 : ui32}> : (
        tensor<2x1x32x32xbf16, #cache>
    ) -> tensor<2x2x32x32xbf16, #rep>

    // permute is transparent — bfp_bf8 propagates.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[PERM:.*]] = "ttnn.permute"(%[[REP]])
    // CHECK-SAME: -> tensor<2x2x32x32x!ttcore.tile<32x32, bfp_bf8>,
    %perm = "ttnn.permute"(%rep) <{permutation = array<i64: 0, 1, 3, 2>}> : (
        tensor<2x2x32x32xbf16, #rep>
    ) -> tensor<2x2x32x32xbf16, #rep>

    // Attention matmul is a stop op: Q is bf16, so allNonCacheAtDtype = false.
    // The matmul receives bfp_bf8 K natively; its result stays bf16. No typecast.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[OUT:.*]] = "ttnn.matmul"(%arg1, %[[PERM]])
    // CHECK-SAME: -> tensor<2x2x1x32xbf16,
    %out = "ttnn.matmul"(%query, %perm) <{transpose_a = false, transpose_b = false}> : (
        tensor<2x2x1x32xbf16, #query>,
        tensor<2x2x32x32xbf16, #rep>
    ) -> tensor<2x2x1x32xbf16, #query>

    return %out : tensor<2x2x1x32xbf16, #query>
  }
}
