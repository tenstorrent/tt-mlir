// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s
// Kimi K2 / DeepSeek V3 TP decode — MLA full bfp_bf8 propagation:
//   c_kv path:  cache → reshape → rms_norm → matmul(W_UK_bfp8) [transparent] → K_absorbed_bfp8
//   kv_pe path: cache → permute → kv_pe_bfp8
//   concat(K_absorbed, kv_pe) is promoted when both inputs are bfp_bf8 (ConcatOp fixpoint)
//   attention matmul(Q_bf16, K_full_bfp8) is a stop op — result stays bf16, no typecast

#dram      = #ttnn.buffer_type<dram>
// c_kv cache: 1 page x 1 head x 32 ctx x 64 latent_dim
#c_kv      = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// kv_pe cache: 1 page x 1 head x 32 ctx x 32 rope_dim
#kv_pe     = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// 2D reshape of c_kv cache: 32 rows x 64 cols
#flat      = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// W_UK weight (up-projection): 64 rows x 32 cols, bfp_bf8
#w_uk      = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>
// matmul result / K_absorbed (2D): 32 x 32
#absorbed  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// K_absorbed reshaped to 4D: 1 x 1 x 32 x 32
#abs4d     = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// concat result: 1 x 1 x 32 x 64
#k_full    = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// K_full permuted (transposed): 1 x 1 x 64 x 32
#k_t       = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// query: 1 x 1 x 1 token x 64 key_dim
#q         = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// attention scores: 1 x 1 x 1 x 32
#scores    = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// gamma for rms_norm: 1D, 64 elements
#gamma     = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// new cache writes: 1 x 1 x 1 token x latent/rope dim
#new_ckv   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#new_pe    = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ptab      = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>

// CHECK-LABEL: func.func @kimi_k2
// Both cache args become bfp_bf8.
// CHECK-SAME: %arg0: tensor<1x1x32x64x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg1: tensor<1x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
module attributes {} {
  func.func @kimi_k2(
      %c_kv_cache:  tensor<1x1x32x64xbf16, #c_kv>  {ttcore.kv_cache},
      %kv_pe_cache: tensor<1x1x32x32xbf16, #kv_pe> {ttcore.kv_cache},
      %query:       tensor<1x1x1x64xbf16, #q>,
      %new_ckv:     tensor<1x1x1x64xbf16, #new_ckv>,
      %new_pe:      tensor<1x1x1x32xbf16, #new_pe>,
      %w_uk:        tensor<64x32x!ttcore.tile<32x32, bfp_bf8>, #w_uk>,
      %gamma:       tensor<64xbf16, #gamma>,
      %ptab:        tensor<1xsi32, #ptab>
  ) -> tensor<1x1x1x32xbf16, #scores> attributes {tt.function_type = "forward_device"} {

    // Write path: no typecasts on paged_update_cache inputs.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.paged_update_cache"(%arg0
    "ttnn.paged_update_cache"(%c_kv_cache, %new_ckv, %ptab) <{share_cache = false}> : (
        tensor<1x1x32x64xbf16, #c_kv>,
        tensor<1x1x1x64xbf16, #new_ckv>,
        tensor<1xsi32, #ptab>
    ) -> ()
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.paged_update_cache"(%arg1
    "ttnn.paged_update_cache"(%kv_pe_cache, %new_pe, %ptab) <{share_cache = false}> : (
        tensor<1x1x32x32xbf16, #kv_pe>,
        tensor<1x1x1x32xbf16, #new_pe>,
        tensor<1xsi32, #ptab>
    ) -> ()

    // c_kv read path: reshape + rms_norm are transparent.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[FLAT:.*]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: -> tensor<32x64x!ttcore.tile<32x32, bfp_bf8>,
    %flat = "ttnn.reshape"(%c_kv_cache) <{shape = [32 : i32, 64 : i32]}> : (
        tensor<1x1x32x64xbf16, #c_kv>
    ) -> tensor<32x64xbf16, #flat>

    // CHECK: %[[NORM:.*]] = "ttnn.rms_norm"(%[[FLAT]], %arg6)
    // CHECK-SAME: -> tensor<32x64x!ttcore.tile<32x32, bfp_bf8>,
    %norm = "ttnn.rms_norm"(%flat, %gamma) <{
        epsilon = 1.0e-7 : f32,
        operandSegmentSizes = array<i32: 1, 1, 0>
    }> : (
        tensor<32x64xbf16, #flat>,
        tensor<64xbf16, #gamma>
    ) -> tensor<32x64xbf16, #flat>

    // matmul(c_kv_norm_bfp8, W_UK_bfp8): all non-cache operands at bfp_bf8 → transparent.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[ABS:.*]] = "ttnn.matmul"(%[[NORM]], %arg5)
    // CHECK-SAME: -> tensor<32x32x!ttcore.tile<32x32, bfp_bf8>,
    %absorbed = "ttnn.matmul"(%norm, %w_uk) <{transpose_a = false, transpose_b = false}> : (
        tensor<32x64xbf16, #flat>,
        tensor<64x32x!ttcore.tile<32x32, bfp_bf8>, #w_uk>
    ) -> tensor<32x32xbf16, #absorbed>

    // Reshape to 4D for concat.
    // CHECK: %[[ABS4:.*]] = "ttnn.reshape"(%[[ABS]])
    // CHECK-SAME: -> tensor<1x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
    %abs4d = "ttnn.reshape"(%absorbed) <{shape = [1 : i32, 1 : i32, 32 : i32, 32 : i32]}> : (
        tensor<32x32xbf16, #absorbed>
    ) -> tensor<1x1x32x32xbf16, #abs4d>

    // kv_pe read path: permute is transparent.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[PE:.*]] = "ttnn.permute"(%arg1)
    // CHECK-SAME: -> tensor<1x1x32x32x!ttcore.tile<32x32, bfp_bf8>,
    %kv_pe_t = "ttnn.permute"(%kv_pe_cache) <{permutation = array<i64: 0, 1, 3, 2>}> : (
        tensor<1x1x32x32xbf16, #kv_pe>
    ) -> tensor<1x1x32x32xbf16, #kv_pe>

    // ConcatOp fixpoint: both inputs now bfp_bf8 → concat is promoted → K_full is bfp_bf8.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[KFULL:.*]] = "ttnn.concat"(%[[ABS4]], %[[PE]])
    // CHECK-SAME: -> tensor<1x1x32x64x!ttcore.tile<32x32, bfp_bf8>,
    %k_full = "ttnn.concat"(%abs4d, %kv_pe_t) <{dim = 3 : si32}> : (
        tensor<1x1x32x32xbf16, #abs4d>,
        tensor<1x1x32x32xbf16, #kv_pe>
    ) -> tensor<1x1x32x64xbf16, #k_full>

    // Transpose K_full for attention matmul.
    // CHECK: %[[KT:.*]] = "ttnn.permute"(%[[KFULL]])
    // CHECK-SAME: -> tensor<1x1x64x32x!ttcore.tile<32x32, bfp_bf8>,
    %k_t = "ttnn.permute"(%k_full) <{permutation = array<i64: 0, 1, 3, 2>}> : (
        tensor<1x1x32x64xbf16, #k_full>
    ) -> tensor<1x1x64x32xbf16, #k_t>

    // Attention matmul: Q is bf16 → stop op; result stays bf16; no typecast.
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: %[[OUT:.*]] = "ttnn.matmul"(%arg2, %[[KT]])
    // CHECK-SAME: -> tensor<1x1x1x32xbf16,
    %scores = "ttnn.matmul"(%query, %k_t) <{transpose_a = false, transpose_b = false}> : (
        tensor<1x1x1x64xbf16, #q>,
        tensor<1x1x64x32xbf16, #k_t>
    ) -> tensor<1x1x1x32xbf16, #scores>

    return %scores : tensor<1x1x1x32xbf16, #scores>
  }
}
