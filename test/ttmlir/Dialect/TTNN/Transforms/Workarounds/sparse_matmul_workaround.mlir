// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Sparsity must be forced to BFloat16 row-major for ttnn.sparse_matmul. The
// compute kernel reads the sparsity buffer through a `uint16_t*` reinterpret
// (see reader_bmm_tile_layout_in0_sender_padding.cpp) and treats any non-zero
// 16-bit value as "valid batch". A 4-byte float32 sparsity makes every other
// E_local read as the low 16 bits of fp32 1.0 = 0x0000 = invalid, so those
// expert outputs are silently zeroed. BFloat16 keeps the element width at 2
// bytes AND preserves non-zero raw bits for any non-zero input value
// (including fractional values like router scores), unlike a value-preserving
// cast to UInt16 which would truncate 0 < |x| < 1 to 0.

// Verify f32 sparsity gets converted to BFloat16 row-major.
module {
  func.func public @test_sparse_matmul_sparsity_f32_to_bf16(
      %a: tensor<1x4x32x256xbf16>,
      %b: tensor<1x4x256x64xbf16>,
      %s: tensor<1x4x1x4xf32>) -> tensor<1x4x1x4x32x64xbf16> {
    // CHECK-LABEL: func.func public @test_sparse_matmul_sparsity_f32_to_bf16
    // CHECK: %[[SPARSITY_BF16:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: tensor<1x4x1x4xf32,
    // CHECK-SAME: -> tensor<1x4x1x4xbf16,
    // CHECK: "ttnn.sparse_matmul"(%{{.*}}, %{{.*}}, %[[SPARSITY_BF16]])
    %0 = "ttir.sparse_matmul"(%a, %b, %s) <{
      is_input_a_sparse = false,
      is_input_b_sparse = true,
      nnz = 0 : i64
    }> : (tensor<1x4x32x256xbf16>, tensor<1x4x256x64xbf16>, tensor<1x4x1x4xf32>)
       -> tensor<1x4x1x4x32x64xbf16>
    return %0 : tensor<1x4x1x4x32x64xbf16>
  }
}

// -----

// UInt16 sparsity (e.g. moe_expert_token_remap reduced output) must also be
// converted to BFloat16 so the kernel sees a 16-bit nonzero value with the
// expected dtype tag. UInt16(1) -> BFloat16(1.0) preserves non-zero-ness.
module {
  func.func public @test_sparse_matmul_sparsity_ui16_to_bf16(
      %a: tensor<1x4x32x256xbf16>,
      %b: tensor<1x4x256x64xbf16>,
      %s: tensor<1x4x1x4xui16>) -> tensor<1x4x1x4x32x64xbf16> {
    // CHECK-LABEL: func.func public @test_sparse_matmul_sparsity_ui16_to_bf16
    // CHECK: %[[SPARSITY_BF16:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: tensor<1x4x1x4xui16,
    // CHECK-SAME: -> tensor<1x4x1x4xbf16,
    // CHECK: "ttnn.sparse_matmul"(%{{.*}}, %{{.*}}, %[[SPARSITY_BF16]])
    %0 = "ttir.sparse_matmul"(%a, %b, %s) <{
      is_input_a_sparse = false,
      is_input_b_sparse = true,
      nnz = 0 : i64
    }> : (tensor<1x4x32x256xbf16>, tensor<1x4x256x64xbf16>, tensor<1x4x1x4xui16>)
       -> tensor<1x4x1x4x32x64xbf16>
    return %0 : tensor<1x4x1x4x32x64xbf16>
  }
}
