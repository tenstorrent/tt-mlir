// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-infer-kv-cache-argument-types --ttir-kv-cache-write-path-norm %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @update_cache_with_rms_norm
  func.func @update_cache_with_rms_norm(
      %cache:   tensor<1x1x32x64xbf16>,
      %new_ckv: tensor<1x1x1x64xbf16>,
      %gamma:   tensor<64xbf16>,
      %w_uk:    tensor<64x32xbf16>
  ) -> tensor<32x32xbf16> attributes {tt.function_type = "forward_device"} {

    // CHECK: [[PRENORM:%.*]] = "ttir.rms_norm"(%arg1, %arg2)
    // CHECK-NEXT: [[CACHE_OUT:%.*]] = "ttir.update_cache"(%arg0, [[PRENORM]]
    %idx = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %cache_out = "ttir.update_cache"(%cache, %new_ckv, %idx) <{batch_offset = 0 : i32}> : (
        tensor<1x1x32x64xbf16>,
        tensor<1x1x1x64xbf16>,
        tensor<1xi32>
    ) -> tensor<1x1x32x64xbf16>

    // CHECK: [[FLAT:%.*]] = "ttir.reshape"([[CACHE_OUT]])
    // CHECK-NOT: "ttir.rms_norm"
    // CHECK: "ttir.matmul"([[FLAT]], %arg3)
    %flat = "ttir.reshape"(%cache_out) <{shape = [32 : i32, 64 : i32]}> : (
        tensor<1x1x32x64xbf16>
    ) -> tensor<32x64xbf16>

    %norm = "ttir.rms_norm"(%flat, %gamma) <{
        normalized_shape = array<i64: 64>,
        epsilon = 1.0e-7 : f32,
        operandSegmentSizes = array<i32: 1, 1, 0>
    }> : (
        tensor<32x64xbf16>,
        tensor<64xbf16>
    ) -> tensor<32x64xbf16>

    %result = "ttir.matmul"(%norm, %w_uk) <{transpose_a = false, transpose_b = false}> : (
        tensor<32x64xbf16>,
        tensor<64x32xbf16>
    ) -> tensor<32x32xbf16>

    return %result : tensor<32x32xbf16>
  }
}

// Prefill pattern: chained fill_cache ops; all chunks get prenorm applied.
module {
  // CHECK-LABEL: func.func @chained_fill_cache_with_rms_norm
  func.func @chained_fill_cache_with_rms_norm(
      %cache:    tensor<1x1x4x64xbf16>,
      %chunk0:   tensor<1x1x2x64xbf16>,
      %chunk1:   tensor<1x1x2x64xbf16>,
      %gamma:    tensor<64xbf16>,
      %w_uk:     tensor<64x32xbf16>
  ) -> tensor<4x32xbf16> attributes {tt.function_type = "forward_device"} {

    // CHECK: [[P0:%.*]] = "ttir.rms_norm"(%arg1, %arg3)
    // CHECK-NEXT: [[R0:%.*]] = "ttir.fill_cache"(%arg0, [[P0]]
    // CHECK: [[P1:%.*]] = "ttir.rms_norm"(%arg2, %arg3)
    // CHECK-NEXT: [[R1:%.*]] = "ttir.fill_cache"([[R0]], [[P1]]
    %r0 = "ttir.fill_cache"(%cache, %chunk0) <{batch_offset = 0 : i32}> : (
        tensor<1x1x4x64xbf16>, tensor<1x1x2x64xbf16>
    ) -> tensor<1x1x4x64xbf16>
    %r1 = "ttir.fill_cache"(%r0, %chunk1) <{batch_offset = 2 : i32}> : (
        tensor<1x1x4x64xbf16>, tensor<1x1x2x64xbf16>
    ) -> tensor<1x1x4x64xbf16>

    // CHECK: [[FLAT:%.*]] = "ttir.reshape"([[R1]])
    // CHECK-NOT: "ttir.rms_norm"
    // CHECK: "ttir.matmul"([[FLAT]], %arg4)
    %flat = "ttir.reshape"(%r1) <{shape = [4 : i32, 64 : i32]}> : (
        tensor<1x1x4x64xbf16>
    ) -> tensor<4x64xbf16>

    %norm = "ttir.rms_norm"(%flat, %gamma) <{
        normalized_shape = array<i64: 64>,
        epsilon = 1.0e-7 : f32,
        operandSegmentSizes = array<i32: 1, 1, 0>
    }> : (
        tensor<4x64xbf16>,
        tensor<64xbf16>
    ) -> tensor<4x64xbf16>

    %result = "ttir.matmul"(%norm, %w_uk) <{transpose_a = false, transpose_b = false}> : (
        tensor<4x64xbf16>,
        tensor<64x32xbf16>
    ) -> tensor<4x32xbf16>

    return %result : tensor<4x32xbf16>
  }
}

module {
  // CHECK-LABEL: func.func @update_cache_no_rms_norm
  func.func @update_cache_no_rms_norm(
      %cache:   tensor<1x1x32x64xbf16>,
      %new_ckv: tensor<1x1x1x64xbf16>,
      %w_uk:    tensor<64x32xbf16>
  ) -> tensor<32x32xbf16> attributes {tt.function_type = "forward_device"} {

    // CHECK: "ttir.update_cache"(%arg0, %arg1
    // CHECK-NOT: "ttir.rms_norm"
    // CHECK: "ttir.matmul"
    %idx = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %cache_out = "ttir.update_cache"(%cache, %new_ckv, %idx) <{batch_offset = 0 : i32}> : (
        tensor<1x1x32x64xbf16>,
        tensor<1x1x1x64xbf16>,
        tensor<1xi32>
    ) -> tensor<1x1x32x64xbf16>

    %flat = "ttir.reshape"(%cache_out) <{shape = [32 : i32, 64 : i32]}> : (
        tensor<1x1x32x64xbf16>
    ) -> tensor<32x64xbf16>

    %result = "ttir.matmul"(%flat, %w_uk) <{transpose_a = false, transpose_b = false}> : (
        tensor<32x64xbf16>,
        tensor<64x32xbf16>
    ) -> tensor<32x32xbf16>

    return %result : tensor<32x32xbf16>
  }
}
