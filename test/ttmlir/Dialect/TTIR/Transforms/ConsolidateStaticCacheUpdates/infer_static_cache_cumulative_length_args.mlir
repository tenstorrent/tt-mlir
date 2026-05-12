// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-infer-static-cache-cumulative-length-args %s | FileCheck %s

// Direct: arg feeds update_cache.update_index. Should be marked.
module {
  // CHECK-LABEL: func.func @mark_direct
  // CHECK-SAME: %arg0: tensor<1x8x16x128xbf16>
  // CHECK-SAME: %arg2: tensor<1xui32> {ttcore.cumulative_length}
  func.func @mark_direct(
      %cache: tensor<1x8x16x128xbf16>,
      %input: tensor<1x8x1x128xbf16>,
      %cumlen: tensor<1xui32>
  ) -> tensor<1x8x16x128xbf16> {
    %0 = "ttir.update_cache"(%cache, %input, %cumlen) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xui32>) -> tensor<1x8x16x128xbf16>
    return %0 : tensor<1x8x16x128xbf16>
  }
}

// Through an add (the typical cumulative_length + arange pattern). The
// cumulative_length arg should be marked; the arange-derived value should
// not produce any marking.
module {
  // CHECK-LABEL: func.func @mark_through_add
  // CHECK-SAME: %arg2: tensor<1xui32> {ttcore.cumulative_length}
  func.func @mark_through_add(
      %cache: tensor<1x8x16x128xbf16>,
      %input: tensor<1x8x1x128xbf16>,
      %cumlen: tensor<1xui32>
  ) -> tensor<1x8x16x128xbf16> {
    %offset = "ttir.full"() <{shape = array<i32: 1>, fill_value = 0 : i32}> : () -> tensor<1xui32>
    %pos = "ttir.add"(%cumlen, %offset) : (tensor<1xui32>, tensor<1xui32>) -> tensor<1xui32>
    %0 = "ttir.update_cache"(%cache, %input, %pos) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xui32>) -> tensor<1x8x16x128xbf16>
    return %0 : tensor<1x8x16x128xbf16>
  }
}

// Negative: the arg is the cache tensor itself, not the update index. The
// cache arg should NOT receive the cumulative_length attribute.
module {
  // CHECK-LABEL: func.func @cache_arg_not_marked
  // CHECK-NOT: ttcore.cumulative_length
  func.func @cache_arg_not_marked(
      %cache: tensor<1x8x16x128xbf16>,
      %input: tensor<1x8x1x128xbf16>
  ) -> tensor<1x8x16x128xbf16> {
    %idx = "ttir.full"() <{shape = array<i32: 1>, fill_value = 0 : i32}> : () -> tensor<1xui32>
    %0 = "ttir.update_cache"(%cache, %input, %idx) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xui32>) -> tensor<1x8x16x128xbf16>
    return %0 : tensor<1x8x16x128xbf16>
  }
}
