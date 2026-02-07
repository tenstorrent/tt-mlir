// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-infer-kv-cache-argument-types %s | FileCheck %s

// Test: update_cache marks cache argument with kv_cache type
module {
  // CHECK-LABEL: func.func @update_cache_marks_kv_cache
  // CHECK-SAME: tensor<1x32x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<kv_cache>}
  // CHECK-SAME: tensor<1x32x1x512xbf16>)
  func.func @update_cache_marks_kv_cache(
      %cache: tensor<1x32x64x512xbf16>,
      %input: tensor<1x32x1x512xbf16>
  ) -> tensor<1x32x64x512xbf16> {
    %idx = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = "ttir.update_cache"(%cache, %input, %idx) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> tensor<1x32x64x512xbf16>
    return %0 : tensor<1x32x64x512xbf16>
  }
}

// Test: fill_cache marks cache argument with kv_cache type
module {
  // CHECK-LABEL: func.func @fill_cache_marks_kv_cache
  // CHECK-SAME: tensor<1x32x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<kv_cache>}
  // CHECK-SAME: tensor<1x32x64x512xbf16>)
  func.func @fill_cache_marks_kv_cache(
      %cache: tensor<1x32x64x512xbf16>,
      %input: tensor<1x32x64x512xbf16>
  ) -> tensor<1x32x64x512xbf16> {
    %0 = "ttir.fill_cache"(%cache, %input) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>) -> tensor<1x32x64x512xbf16>
    return %0 : tensor<1x32x64x512xbf16>
  }
}

// Test: chained cache ops trace back to original argument
module {
  // CHECK-LABEL: func.func @chained_cache_ops
  // CHECK-SAME: tensor<1x32x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<kv_cache>}
  func.func @chained_cache_ops(
      %cache: tensor<1x32x64x512xbf16>,
      %input1: tensor<1x32x64x512xbf16>,
      %input2: tensor<1x32x1x512xbf16>
  ) -> tensor<1x32x64x512xbf16> {
    %filled = "ttir.fill_cache"(%cache, %input1) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>) -> tensor<1x32x64x512xbf16>
    %idx = "ttir.constant"() <{value = dense<63> : tensor<1xi32>}> : () -> tensor<1xi32>
    %updated = "ttir.update_cache"(%filled, %input2, %idx) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> tensor<1x32x64x512xbf16>
    return %updated : tensor<1x32x64x512xbf16>
  }
}

// Test: multiple caches in same function - both get marked
module {
  // CHECK-LABEL: func.func @multiple_caches
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<kv_cache>}
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<kv_cache>}
  func.func @multiple_caches(
      %key_cache: tensor<1x32x64x512xbf16>,
      %value_cache: tensor<1x32x64x512xbf16>,
      %key_input: tensor<1x32x1x512xbf16>,
      %value_input: tensor<1x32x1x512xbf16>
  ) -> (tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>) {
    %idx = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %key_updated = "ttir.update_cache"(%key_cache, %key_input, %idx) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> tensor<1x32x64x512xbf16>
    %value_updated = "ttir.update_cache"(%value_cache, %value_input, %idx) <{batch_offset = 0: i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> tensor<1x32x64x512xbf16>
    return %key_updated, %value_updated : tensor<1x32x64x512xbf16>, tensor<1x32x64x512xbf16>
  }
}

// Test: non-cache arguments are not marked
module {
  // CHECK-LABEL: func.func @non_cache_not_marked
  // CHECK-NOT: ttcore.argument_type
  func.func @non_cache_not_marked(
      %input: tensor<1x32x64x512xbf16>
  ) -> tensor<1x32x64x512xbf16> {
    return %input : tensor<1x32x64x512xbf16>
  }
}
