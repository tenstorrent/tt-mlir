// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-layout --ttnn-workaround --canonicalize %s | FileCheck %s

// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test sparse_matmul tile dims workaround.
// When input A has M=1 (untiled), the workaround should insert
// reshape/permute to tile to M=32 and untile the output.

module @test_sparse_matmul_tile_dims attributes {} {

  // Gate-up: untiled input [BD=2, S=128, M=1, H=2880]
  // Workaround should tile to [BD=2, S/M=4, M=32, H=2880]
  func.func public @sparse_matmul_gate_up_untiled(
    %input: tensor<2x128x1x2880xbf16>,
    %weight: tensor<1x4x2880x5760xbf16>,
    %sparsity: tensor<1x1x8x4xbf16>
  ) -> tensor<2x128x1x4x1x5760xbf16> {
    // CHECK-LABEL: func.func public @sparse_matmul_gate_up_untiled
    // Verify tiled sparse_matmul with M=32
    // CHECK: "ttnn.sparse_matmul"
    // CHECK-SAME: tensor<2x4x32x2880xbf16
    // CHECK-SAME: tensor<1x4x2880x5760xbf16
    // CHECK-SAME: tensor<2x4x1x4xbf16
    // CHECK-SAME: -> tensor<2x4x1x4x32x5760xbf16
    // Verify legacy gate-up factoring is preserved before re-expanding.
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [2 : i32, 4 : i32, 4 : i32, 32 : i32, 5760 : i32]}>
    // CHECK: "ttnn.permute"(%{{.*}}) <{permutation = array<i64: 0, 1, 3, 2, 4>}>
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [2 : i32, 128 : i32, 1 : i32, 4 : i32, 1 : i32, 5760 : i32]}>
    %result = "ttnn.sparse_matmul"(%input, %weight, %sparsity) <{
      is_input_a_sparse = false,
      is_input_b_sparse = true,
      nnz = 0 : i64
    }> : (tensor<2x128x1x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<1x1x8x4xbf16>) -> tensor<2x128x1x4x1x5760xbf16>
    return %result : tensor<2x128x1x4x1x5760xbf16>
  }

  // Down: untiled input [BD*S=256, E=4, M=1, inter=2880]
  // Workaround should tile to [BD*S/M=8, E=4, M=32, inter=2880]
  func.func public @sparse_matmul_down_untiled(
    %input: tensor<256x4x1x2880xbf16>,
    %weight: tensor<1x4x2880x2880xbf16>,
    %sparsity: tensor<1x1x8x4xbf16>
  ) -> tensor<256x4x1x2880xbf16> {
    // CHECK-LABEL: func.func public @sparse_matmul_down_untiled
    // Verify tiled sparse_matmul with M=32
    // CHECK: "ttnn.sparse_matmul"
    // CHECK-SAME: tensor<8x4x32x2880xbf16
    // CHECK-SAME: tensor<1x4x2880x2880xbf16
    // CHECK-SAME: -> tensor<8x4x32x2880xbf16
    %result = "ttnn.sparse_matmul"(%input, %weight, %sparsity) <{
      is_input_a_sparse = true,
      is_input_b_sparse = false,
      nnz = 0 : i64
    }> : (tensor<256x4x1x2880xbf16>, tensor<1x4x2880x2880xbf16>, tensor<1x1x8x4xbf16>) -> tensor<256x4x1x2880xbf16>
    return %result : tensor<256x4x1x2880xbf16>
  }

  // Already tiled: M=32, workaround should NOT apply
  func.func public @sparse_matmul_already_tiled(
    %input: tensor<2x4x32x2880xbf16>,
    %weight: tensor<1x4x2880x5760xbf16>,
    %sparsity: tensor<2x4x1x4xbf16>
  ) -> tensor<2x4x1x4x32x5760xbf16> {
    // CHECK-LABEL: func.func public @sparse_matmul_already_tiled
    // CHECK: "ttnn.sparse_matmul"
    // CHECK-SAME: tensor<2x4x32x2880xbf16
    // No extra reshape/permute should be inserted
    %result = "ttnn.sparse_matmul"(%input, %weight, %sparsity) <{
      is_input_a_sparse = false,
      is_input_b_sparse = true,
      nnz = 0 : i64
    }> : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>, tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
    return %result : tensor<2x4x1x4x32x5760xbf16>
  }

  // Already tiled gate-up + flattened SwiGLU chain:
  // rewrite should restore M-preserving activation layout.
  func.func public @sparse_matmul_already_tiled_gate_up_activation(
    %input: tensor<2x4x32x2880xbf16>,
    %weight: tensor<1x1x2880x5760xbf16>,
    %sparsity: tensor<2x4x1x1xbf16>,
    %up_bias: tensor<1x1x1x5760xbf16>,
    %scalar_bias: tensor<1x1x1x1xbf16>,
    %scalar_mul: tensor<1x1x1x1xbf16>
  ) -> tensor<8x1x32x2880xbf16> {
    // CHECK-LABEL: func.func public @sparse_matmul_already_tiled_gate_up_activation
    // CHECK-NOT: tensor<256x1x1x5760xbf16
    // CHECK: "ttnn.add"(%{{.*}}, %{{.*}}) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<2x4x1x32x5760xbf16
    // CHECK: "ttnn.permute"(%{{.*}}) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x32x1x5760xbf16
    // CHECK: "ttnn.slice_static"(%{{.*}}) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [8 : i32, 1 : i32, 32 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}>
    %sm = "ttnn.sparse_matmul"(%input, %weight, %sparsity) <{
      is_input_a_sparse = false,
      is_input_b_sparse = true,
      nnz = 0 : i64
    }> : (tensor<2x4x32x2880xbf16>, tensor<1x1x2880x5760xbf16>, tensor<2x4x1x1xbf16>) -> tensor<2x4x1x1x32x5760xbf16>

    %r0 = "ttnn.reshape"(%sm) <{shape = [2 : i32, 4 : i32, 1 : i32, 32 : i32, 5760 : i32]}>
      : (tensor<2x4x1x1x32x5760xbf16>) -> tensor<2x4x1x32x5760xbf16>
    %p0 = "ttnn.permute"(%r0) <{permutation = array<i64: 0, 1, 3, 2, 4>}>
      : (tensor<2x4x1x32x5760xbf16>) -> tensor<2x4x32x1x5760xbf16>
    %flat = "ttnn.reshape"(%p0) <{shape = [256 : i32, 1 : i32, 1 : i32, 5760 : i32]}>
      : (tensor<2x4x32x1x5760xbf16>) -> tensor<256x1x1x5760xbf16>

    %pre = "ttnn.add"(%flat, %up_bias) <{dtype = #ttcore.supportedDataTypes<bf16>}>
      : (tensor<256x1x1x5760xbf16>, tensor<1x1x1x5760xbf16>) -> tensor<256x1x1x5760xbf16>

    %gate = "ttnn.slice_static"(%pre) <{
      begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32],
      ends = [256 : i32, 1 : i32, 1 : i32, 5760 : i32],
      step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]
    }> : (tensor<256x1x1x5760xbf16>) -> tensor<256x1x1x2880xbf16>
    %gate_clamped = "ttnn.clamp_scalar"(%gate) <{max = 7.000000e+00 : f32, min = -7.000000e+00 : f32}>
      : (tensor<256x1x1x2880xbf16>) -> tensor<256x1x1x2880xbf16>
    %gate_added = "ttnn.add"(%gate_clamped, %scalar_bias) <{dtype = #ttcore.supportedDataTypes<bf16>}>
      : (tensor<256x1x1x2880xbf16>, tensor<1x1x1x1xbf16>) -> tensor<256x1x1x2880xbf16>

    %value = "ttnn.slice_static"(%pre) <{
      begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
      ends = [256 : i32, 1 : i32, 1 : i32, 5760 : i32],
      step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]
    }> : (tensor<256x1x1x5760xbf16>) -> tensor<256x1x1x2880xbf16>
    %value_clamped = "ttnn.clamp_scalar"(%value) <{max = 7.000000e+00 : f32, min = 0xFF800000 : f32}>
      : (tensor<256x1x1x2880xbf16>) -> tensor<256x1x1x2880xbf16>
    %value_scaled = "ttnn.multiply"(%value_clamped, %scalar_mul) <{dtype = #ttcore.supportedDataTypes<bf16>}>
      : (tensor<256x1x1x2880xbf16>, tensor<1x1x1x1xbf16>) -> tensor<256x1x1x2880xbf16>
    %value_sigmoid = "ttnn.sigmoid"(%value_scaled)
      : (tensor<256x1x1x2880xbf16>) -> tensor<256x1x1x2880xbf16>
    %value_gated = "ttnn.multiply"(%value_clamped, %value_sigmoid) <{dtype = #ttcore.supportedDataTypes<bf16>}>
      : (tensor<256x1x1x2880xbf16>, tensor<256x1x1x2880xbf16>) -> tensor<256x1x1x2880xbf16>

    %out = "ttnn.multiply"(%gate_added, %value_gated) <{dtype = #ttcore.supportedDataTypes<bf16>}>
      : (tensor<256x1x1x2880xbf16>, tensor<256x1x1x2880xbf16>) -> tensor<256x1x1x2880xbf16>
    %r1 = "ttnn.reshape"(%out) <{shape = [8 : i32, 32 : i32, 1 : i32, 2880 : i32]}>
      : (tensor<256x1x1x2880xbf16>) -> tensor<8x32x1x2880xbf16>
    %p1 = "ttnn.permute"(%r1) <{permutation = array<i64: 0, 2, 1, 3>}>
      : (tensor<8x32x1x2880xbf16>) -> tensor<8x1x32x2880xbf16>
    return %p1 : tensor<8x1x32x2880xbf16>
  }
}
