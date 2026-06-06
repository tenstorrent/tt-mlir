// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tests for FillCacheInputPadRewritePattern (see header for rationale).
// Refs: tt-metal#42779, tt-xla#4785.

#dram = #ttnn.buffer_type<dram>
#input_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 18 + d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#cache_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Case 1: unaligned input seq_len (18) -> pad to 32, then fill_cache.
module @case1_fill_cache_unaligned attributes {} {
  func.func public @fill_cache_unaligned_seq(
      %cache: tensor<1x32x64x64xbf16, #cache_layout>,
      %input: tensor<1x32x18x64xbf16, #input_layout>) {
    // CHECK-LABEL: func.func public @fill_cache_unaligned_seq
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

// Case 2: input seq_len already tile-aligned (32) -> no pad inserted.
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

// Case 3: paged_fill_cache with unaligned input -> same pad as Case 1.
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

// -----

#dram = #ttnn.buffer_type<dram>
#cache_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2016 + d1 * 63 + d2, d3), <1x1>, memref<63x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#input_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2016 + d1 * 63 + d2, d3), <1x1>, memref<63x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Case 4: cache also non-tile-aligned (63) -> pattern bails (padding input
// to 64 would violate input.dim(-2) <= cache.dim(-2)).
module @case4_fill_cache_cache_also_unaligned attributes {} {
  func.func public @fill_cache_cache_also_unaligned(
      %cache: tensor<1x32x63x511xbf16, #cache_layout>,
      %input: tensor<1x32x63x511xbf16, #input_layout>) {
    // CHECK-LABEL: func.func public @fill_cache_cache_also_unaligned
    // CHECK-NOT: "ttnn.pad"
    // CHECK: "ttnn.fill_cache"(%arg0, %arg1)
    "ttnn.fill_cache"(%cache, %input) <{batch_offset = 0 : i32}> :
        (tensor<1x32x63x511xbf16, #cache_layout>,
         tensor<1x32x63x511xbf16, #input_layout>) -> ()
    return
  }
}
