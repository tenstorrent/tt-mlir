// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// EmbeddingOpPadHiddenDimRewritePattern triggers when the weight's last dim
// is in the regression-prone region:
//   PASS iff  hidden_dim < 8192  OR  hidden_dim % 2048 == 0
// On a hit, the weight is high-padded on the last dim up to the next
// multiple of 2048, ttnn.embedding runs on the padded weight, and the
// result is sliced back to the original hidden dim.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_input = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>

// Bad case: D=10752, tile_count=336. Pads to 12288 (=6*2048, tile_count=384).
#ttnn_layout_weight_bad = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x336x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_result_bad = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2x336x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Good case (large): D=10240, tile_count=320. Multiple of 2048 already; no rewrite.
#ttnn_layout_weight_large_ok = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x320x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_result_large_ok = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2x320x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Good case (small): D=2560, tile_count=80. Below the 8192 threshold; no
// rewrite even though tile_count is non-pow2.
#ttnn_layout_weight_small_ok = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x80x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_result_small_ok = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2x80x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  // Bad case: rewrite expected.
  // CHECK-LABEL: func.func @bad_d10752
  func.func @bad_d10752(
      %arg0: tensor<1x64xui32, #ttnn_layout_input>,
      %arg1: tensor<256x10752xbf16, #ttnn_layout_weight_bad>)
      -> tensor<1x64x10752xbf16, #ttnn_layout_result_bad> {
    // Weight is high-padded by 1536 on the last dim up to 12288, the
    // embedding runs on the padded weight, and the result is sliced back
    // to 10752 on the last dim. Layout-conversion ops ("ttnn.to_layout")
    // can sit between these in any order, so we only check that each step
    // is present with the right attribute / shape.
    // CHECK: ttnn.pad
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 1536>
    // CHECK-SAME: -> tensor<256x12288xbf16
    // CHECK: ttnn.embedding
    // CHECK-SAME: -> tensor<1x64x12288xbf16
    // CHECK: ttnn.slice_static
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 64 : i32, 10752 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 1 : i32]
    // CHECK-SAME: -> tensor<1x64x10752xbf16
    %0 = "ttnn.embedding"(%arg0, %arg1) : (tensor<1x64xui32, #ttnn_layout_input>, tensor<256x10752xbf16, #ttnn_layout_weight_bad>) -> tensor<1x64x10752xbf16, #ttnn_layout_result_bad>
    return %0 : tensor<1x64x10752xbf16, #ttnn_layout_result_bad>
  }

  // Good case (large, multiple of 2048): pattern must NOT fire.
  // CHECK-LABEL: func.func @good_d10240_multiple_of_2048
  func.func @good_d10240_multiple_of_2048(
      %arg0: tensor<1x64xui32, #ttnn_layout_input>,
      %arg1: tensor<256x10240xbf16, #ttnn_layout_weight_large_ok>)
      -> tensor<1x64x10240xbf16, #ttnn_layout_result_large_ok> {
    // CHECK-NOT: ttnn.pad
    // CHECK-NOT: ttnn.slice_static
    // CHECK: ttnn.embedding
    // CHECK-SAME: -> tensor<1x64x10240xbf16
    %0 = "ttnn.embedding"(%arg0, %arg1) : (tensor<1x64xui32, #ttnn_layout_input>, tensor<256x10240xbf16, #ttnn_layout_weight_large_ok>) -> tensor<1x64x10240xbf16, #ttnn_layout_result_large_ok>
    return %0 : tensor<1x64x10240xbf16, #ttnn_layout_result_large_ok>
  }

  // Good case (small, below 8192 threshold): pattern must NOT fire even
  // though tile_count=80 is non-power-of-two. Below 8192 every shape we
  // measured passes regardless of factorization.
  // CHECK-LABEL: func.func @good_d2560_below_threshold
  func.func @good_d2560_below_threshold(
      %arg0: tensor<1x64xui32, #ttnn_layout_input>,
      %arg1: tensor<256x2560xbf16, #ttnn_layout_weight_small_ok>)
      -> tensor<1x64x2560xbf16, #ttnn_layout_result_small_ok> {
    // CHECK-NOT: ttnn.pad
    // CHECK-NOT: ttnn.slice_static
    // CHECK: ttnn.embedding
    // CHECK-SAME: -> tensor<1x64x2560xbf16
    %0 = "ttnn.embedding"(%arg0, %arg1) : (tensor<1x64xui32, #ttnn_layout_input>, tensor<256x2560xbf16, #ttnn_layout_weight_small_ok>) -> tensor<1x64x2560xbf16, #ttnn_layout_result_small_ok>
    return %0 : tensor<1x64x2560xbf16, #ttnn_layout_result_small_ok>
  }
}
