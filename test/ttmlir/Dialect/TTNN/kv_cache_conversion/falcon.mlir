// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s
// Falcon3 7B prefill — fill_cache write path: bf16 caches → bfp_bf8; a typecast is
// inserted on the fill value (bf16 → bfp_bf8) because FillCacheOp is not excluded.

#dram   = #ttnn.buffer_type<dram>
// K/V cache: 2 batch x 1 head x 32 seq x 32 head_dim
#cache  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Prefill fill value: 1 batch x 1 head x 32 seq x 32 head_dim
#fill   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @falcon
// Cache args become bfp_bf8; fill tensors stay bf16.
// CHECK-SAME: %arg0: tensor<2x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg1: tensor<2x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg2: tensor<1x1x32x32xbf16,
// CHECK-SAME: %arg3: tensor<1x1x32x32xbf16,
module attributes {} {
  func.func @falcon(
      %k_cache: tensor<2x1x32x32xbf16, #cache> {ttcore.kv_cache},
      %v_cache: tensor<2x1x32x32xbf16, #cache> {ttcore.kv_cache},
      %new_k:   tensor<1x1x32x32xbf16, #fill>,
      %new_v:   tensor<1x1x32x32xbf16, #fill>
  ) attributes {tt.function_type = "forward_device"} {
    // fill_cache: typecast inserted on the fill value (bf16 → bfp_bf8).
    // CHECK: %[[CK:.*]] = "ttnn.typecast"(%arg2)
    // CHECK-SAME: -> tensor<1x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK: "ttnn.fill_cache"(%arg0, %[[CK]])
    "ttnn.fill_cache"(%k_cache, %new_k) <{batch_offset = 0 : i32}> : (
        tensor<2x1x32x32xbf16, #cache>,
        tensor<1x1x32x32xbf16, #fill>
    ) -> ()
    // CHECK: %[[CV:.*]] = "ttnn.typecast"(%arg3)
    // CHECK-SAME: -> tensor<1x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK: "ttnn.fill_cache"(%arg1, %[[CV]])
    "ttnn.fill_cache"(%v_cache, %new_v) <{batch_offset = 0 : i32}> : (
        tensor<2x1x32x32xbf16, #cache>,
        tensor<1x1x32x32xbf16, #fill>
    ) -> ()
    return
  }
}
