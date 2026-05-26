// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Workaround under test:
// `FillCacheOpInputPadRewritePattern` / `PagedFillCacheOpInputPadRewritePattern`
// zero-pad the cache-write input's seq_len (dim -2) up to the next tile
// multiple. This neutralizes any garbage (+/-Inf / NaN) sitting in the
// implicit tile-pad rows of upstream producers, which the cache-write
// kernel otherwise iterates over and copies verbatim into the KV cache
// (tt-metal#42779, tt-xla#4785).

#dram = #ttnn.buffer_type<dram>
#input_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 18 + d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#cache_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Case 1: fill_cache with non-tile-aligned input seq_len (18). The pattern
// MUST insert a `ttnn.pad` that right-pads dim -2 from 18 up to 32 with
// zeros, then rewrite fill_cache to consume the padded input.
module @case1_fill_cache_unaligned attributes {} {
  func.func public @fill_cache_unaligned_seq(
      %cache: tensor<1x32x64x64xbf16, #cache_layout>,
      %input: tensor<1x32x18x64xbf16, #input_layout>) {
    // CHECK-LABEL: func.func public @fill_cache_unaligned_seq
    // padding layout for 4D is
    // [d0_lo, d0_hi, d1_lo, d1_hi, d2_lo, d2_hi, d3_lo, d3_hi],
    // and we right-pad dim 2 by 14 (18 -> 32).
    // CHECK: %[[PADDED:.*]] = "ttnn.pad"(%arg1)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 14, 0, 0>
    // CHECK-SAME: value = 0.000000e+00
    // CHECK: "ttnn.fill_cache"(%arg0, %[[PADDED]])
    // CHECK-SAME: batch_offset = 0
    "ttnn.fill_cache"(%cache, %input) <{batch_offset = 0 : i32}> :
        (tensor<1x32x64x64xbf16, #cache_layout>,
         tensor<1x32x18x64xbf16, #input_layout>) -> ()
    return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#input_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#cache_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Case 2: fill_cache whose input seq_len is already tile-aligned (32).
// The pattern MUST NOT insert a pad — the original op survives unchanged.
module @case2_fill_cache_aligned attributes {} {
  func.func public @fill_cache_aligned_seq(
      %cache: tensor<1x32x64x64xbf16, #cache_layout>,
      %input: tensor<1x32x32x64xbf16, #input_layout>) {
    // CHECK-LABEL: func.func public @fill_cache_aligned_seq
    // CHECK-NOT: "ttnn.pad"
    // CHECK: "ttnn.fill_cache"(%arg0, %arg1)
    "ttnn.fill_cache"(%cache, %input) <{batch_offset = 0 : i32}> :
        (tensor<1x32x64x64xbf16, #cache_layout>,
         tensor<1x32x32x64xbf16, #input_layout>) -> ()
    return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#input_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 18 + d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#cache_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#page_table_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2xi32, #dram>, <interleaved>>

// Case 3: paged_fill_cache with non-tile-aligned input seq_len. Same
// hazard, same scrub.
module @case3_paged_fill_cache_unaligned attributes {} {
  func.func public @paged_fill_cache_unaligned_seq(
      %cache: tensor<1x32x64x64xbf16, #cache_layout>,
      %input: tensor<1x32x18x64xbf16, #input_layout>,
      %page_table: tensor<1x2xi32, #page_table_layout>) {
    // CHECK-LABEL: func.func public @paged_fill_cache_unaligned_seq
    // CHECK: %[[PADDED:.*]] = "ttnn.pad"(%arg1)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 14, 0, 0>
    // CHECK: "ttnn.paged_fill_cache"(%arg0, %[[PADDED]], %arg2)
    "ttnn.paged_fill_cache"(%cache, %input, %page_table) :
        (tensor<1x32x64x64xbf16, #cache_layout>,
         tensor<1x32x18x64xbf16, #input_layout>,
         tensor<1x2xi32, #page_table_layout>) -> ()
    return
  }
}
