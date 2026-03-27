// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround --canonicalize %s | FileCheck %s

// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test bundled MoE workarounds.
// Non-canonical rank/layout forms should be canonicalized to HW-required
// shapes here, not in conversion patterns.
// Decode-like all_to_all_combine with output_shard_dim=2 and S=1 should be
// rewritten to output_shard_dim=1 + reshape back to the original result shape.
// Gate-up sparse_matmul activation chains should also be rewritten here to
// preserve the M dimension through the pointwise subgraph.

module @test_moe_workaround attributes {} {

  func.func public @all_to_all_dispatch_rank3(
    %input: tensor<64x128x7168xbf16>,
    %indices: tensor<8192x8xi64>,
    %mapping: tensor<1x1x256x8xi64>
  ) -> (tensor<1x64x128x7168xbf16>, tensor<1x64x128x8xi64>) {
    // CHECK-LABEL: func.func public @all_to_all_dispatch_rank3
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [64 : i32, 1 : i32, 128 : i32, 7168 : i32]}>
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [64 : i32, 1 : i32, 128 : i32, 8 : i32]}>
    // CHECK: "ttnn.all_to_all_dispatch"(%{{.*}}, %{{.*}}, %{{.*}})
    %dispatched, %metadata = "ttnn.all_to_all_dispatch"(%input, %indices, %mapping) <{
      cluster_axis = 0 : i64,
      num_devices = 2 : i64
    }> : (tensor<64x128x7168xbf16>, tensor<8192x8xi64>, tensor<1x1x256x8xi64>) -> (tensor<1x64x128x7168xbf16>, tensor<1x64x128x8xi64>)
    return %dispatched, %metadata : tensor<1x64x128x7168xbf16>, tensor<1x64x128x8xi64>
  }

  func.func public @all_to_all_combine_layout_bdseh(
    %input: tensor<64x128x32x7168xbf16>,
    %metadata: tensor<1x64x128x8xi64>,
    %mapping: tensor<1x1x256x8xi64>
  ) -> tensor<8x8x128x7168xbf16> {
    // CHECK-LABEL: func.func public @all_to_all_combine_layout_bdseh
    // CHECK: "ttnn.permute"(%{{.*}}) <{permutation = array<i64: 2, 0, 1, 3>}>
    // CHECK: "ttnn.all_to_all_combine"(%{{.*}}, %{{.*}}, %{{.*}})
    %result = "ttnn.all_to_all_combine"(%input, %metadata, %mapping) <{
      cluster_axis = 0 : i64,
      num_devices = 8 : i64,
      num_experts_per_tok = 8 : i64,
      output_shard_dim = 1 : i64
    }> : (tensor<64x128x32x7168xbf16>, tensor<1x64x128x8xi64>, tensor<1x1x256x8xi64>) -> tensor<8x8x128x7168xbf16>
    return %result : tensor<8x8x128x7168xbf16>
  }

  func.func public @moe_expert_token_remap_topk_rank2(
    %topk: tensor<1024x8xbf16>,
    %mapping: tensor<1x1x256x8xi64>,
    %metadata: tensor<1x64x128x8xi64>
  ) -> (tensor<1x64x128x8xbf16>, tensor<1x1x8x8xbf16>) {
    // CHECK-LABEL: func.func public @moe_expert_token_remap_topk_rank2
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [8 : i32, 128 : i32, 8 : i32]}>
    // CHECK: "ttnn.concat"(%{{.*}}
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [1 : i32, 64 : i32, 128 : i32, 8 : i32]}>
    // CHECK: "ttnn.moe_expert_token_remap"(%{{.*}}, %{{.*}}, %{{.*}})
    %mapping_out, %reduced = "ttnn.moe_expert_token_remap"(%topk, %mapping, %metadata) <{
      reduction_size = 8 : i64
    }> : (tensor<1024x8xbf16>, tensor<1x1x256x8xi64>, tensor<1x64x128x8xi64>) -> (tensor<1x64x128x8xbf16>, tensor<1x1x8x8xbf16>)
    return %mapping_out, %reduced : tensor<1x64x128x8xbf16>, tensor<1x1x8x8xbf16>
  }

  func.func public @all_to_all_combine_decode(
    %input: tensor<8x64x1x7168xbf16>,
    %metadata: tensor<1x64x1x8xi64>,
    %mapping: tensor<1x1x256x8xi64>
  ) -> tensor<8x8x1x7168xbf16> {
    // CHECK-LABEL: func.func public @all_to_all_combine_decode
    // CHECK: "ttnn.all_to_all_combine"
    // CHECK-SAME: output_shard_dim = 1
    // CHECK-SAME: -> tensor<8x1x8x7168xbf16
    // CHECK: "ttnn.reshape"(%{{.*}}) <{shape = [8 : i32, 8 : i32, 1 : i32, 7168 : i32]}>
    %result = "ttnn.all_to_all_combine"(%input, %metadata, %mapping) <{
      cluster_axis = 0 : i64,
      num_devices = 2 : i64,
      num_experts_per_tok = 8 : i64,
      output_shard_dim = 2 : i64
    }> : (tensor<8x64x1x7168xbf16>, tensor<1x64x1x8xi64>, tensor<1x1x256x8xi64>) -> tensor<8x8x1x7168xbf16>
    return %result : tensor<8x8x1x7168xbf16>
  }

  // Already tiled gate-up chain with a non-multiply merge op.
  // This exercises the generic nearest-common-pointwise-user rewrite.
  func.func public @sparse_matmul_already_tiled_gate_up_add_merge(
    %input: tensor<2x4x32x2880xbf16>,
    %weight: tensor<1x1x2880x5760xbf16>,
    %sparsity: tensor<2x4x1x1xbf16>,
    %up_bias: tensor<1x1x1x5760xbf16>
  ) -> tensor<8x1x32x2880xbf16> {
    // CHECK-LABEL: func.func public @sparse_matmul_already_tiled_gate_up_add_merge
    // CHECK-NOT: tensor<256x1x1x5760xbf16
    // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%{{.*}}) <{shape = [8 : i32, 1 : i32, 32 : i32, 5760 : i32]}>
    // CHECK: %[[PRE:.*]] = "ttnn.add"(%[[RESHAPE]], %{{.*}}) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x1x32x5760xbf16>, tensor<1x1x1x5760xbf16>) -> tensor<8x1x32x5760xbf16>
    // CHECK-DAG: %[[GATE:.*]] = "ttnn.slice_static"(%[[PRE]]) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [8 : i32, 1 : i32, 32 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}>
    // CHECK-DAG: %[[VALUE:.*]] = "ttnn.slice_static"(%[[PRE]]) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 1 : i32, 32 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}>
    // CHECK-DAG: %[[GATE_SILU:.*]] = "ttnn.silu"(%[[GATE]]) : (tensor<8x1x32x2880xbf16>) -> tensor<8x1x32x2880xbf16>
    // CHECK-DAG: %[[VALUE_SIGMOID:.*]] = "ttnn.sigmoid"(%[[VALUE]]) : (tensor<8x1x32x2880xbf16>) -> tensor<8x1x32x2880xbf16>
    // CHECK: "ttnn.add"(%[[GATE_SILU]], %[[VALUE_SIGMOID]]) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x1x32x2880xbf16>, tensor<8x1x32x2880xbf16>) -> tensor<8x1x32x2880xbf16>
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
    %gate_silu = "ttnn.silu"(%gate)
      : (tensor<256x1x1x2880xbf16>) -> tensor<256x1x1x2880xbf16>

    %value = "ttnn.slice_static"(%pre) <{
      begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
      ends = [256 : i32, 1 : i32, 1 : i32, 5760 : i32],
      step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]
    }> : (tensor<256x1x1x5760xbf16>) -> tensor<256x1x1x2880xbf16>
    %value_sigmoid = "ttnn.sigmoid"(%value)
      : (tensor<256x1x1x2880xbf16>) -> tensor<256x1x1x2880xbf16>

    %out = "ttnn.add"(%gate_silu, %value_sigmoid) <{dtype = #ttcore.supportedDataTypes<bf16>}>
      : (tensor<256x1x1x2880xbf16>, tensor<256x1x1x2880xbf16>) -> tensor<256x1x1x2880xbf16>
    %r1 = "ttnn.reshape"(%out) <{shape = [8 : i32, 32 : i32, 1 : i32, 2880 : i32]}>
      : (tensor<256x1x1x2880xbf16>) -> tensor<8x32x1x2880xbf16>
    %p1 = "ttnn.permute"(%r1) <{permutation = array<i64: 0, 2, 1, 3>}>
      : (tensor<8x32x1x2880xbf16>) -> tensor<8x1x32x2880xbf16>
    return %p1 : tensor<8x1x32x2880xbf16>
  }

  // Untiled gate-up chain should be rewritten directly from the tiled sparse
  // output instead of going through the legacy [BD, S, E, N] activation path.
  func.func public @sparse_matmul_gate_up_untiled_add_merge(
    %input: tensor<2x128x1x2880xbf16>,
    %weight: tensor<1x1x2880x5760xbf16>,
    %sparsity: tensor<1x1x8x1xbf16>,
    %up_bias: tensor<1x1x1x5760xbf16>
  ) -> tensor<8x1x32x2880xbf16> {
    // CHECK-LABEL: func.func public @sparse_matmul_gate_up_untiled_add_merge
    // CHECK: "ttnn.sparse_matmul"
    // CHECK-SAME: tensor<2x4x32x2880xbf16
    // CHECK-SAME: tensor<1x1x2880x5760xbf16
    // CHECK-SAME: tensor<2x4x1x1xbf16
    // CHECK-SAME: -> tensor<2x4x1x1x32x5760xbf16
    // CHECK-NOT: tensor<2x128x1x5760xbf16
    // CHECK: %[[PRE:.*]] = "ttnn.add"(%{{.*}}, %{{.*}}) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x1x32x5760xbf16>, tensor<1x1x1x5760xbf16>) -> tensor<8x1x32x5760xbf16>
    // CHECK-DAG: %[[UGATE:.*]] = "ttnn.slice_static"(%[[PRE]]) <{begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ends = [8 : i32, 1 : i32, 32 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}>
    // CHECK-DAG: %[[UVALUE:.*]] = "ttnn.slice_static"(%[[PRE]]) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 1 : i32, 32 : i32, 5760 : i32], step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]}>
    // CHECK-DAG: %[[UGATE_SILU:.*]] = "ttnn.silu"(%[[UGATE]]) : (tensor<8x1x32x2880xbf16>) -> tensor<8x1x32x2880xbf16>
    // CHECK-DAG: %[[UVALUE_SIGMOID:.*]] = "ttnn.sigmoid"(%[[UVALUE]]) : (tensor<8x1x32x2880xbf16>) -> tensor<8x1x32x2880xbf16>
    // CHECK: %[[UMERGE:.*]] = "ttnn.add"(%[[UGATE_SILU]], %[[UVALUE_SIGMOID]]) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<8x1x32x2880xbf16>, tensor<8x1x32x2880xbf16>) -> tensor<8x1x32x2880xbf16>
    // CHECK: "ttnn.permute"(%[[UMERGE]]) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<8x1x32x2880xbf16>) -> tensor<8x32x1x2880xbf16>
    %sm = "ttnn.sparse_matmul"(%input, %weight, %sparsity) <{
      is_input_a_sparse = false,
      is_input_b_sparse = true,
      nnz = 0 : i64
    }> : (tensor<2x128x1x2880xbf16>, tensor<1x1x2880x5760xbf16>, tensor<1x1x8x1xbf16>) -> tensor<2x128x1x1x1x5760xbf16>

    %r0 = "ttnn.reshape"(%sm) <{shape = [2 : i32, 128 : i32, 1 : i32, 5760 : i32]}>
      : (tensor<2x128x1x1x1x5760xbf16>) -> tensor<2x128x1x5760xbf16>
    %pre = "ttnn.add"(%r0, %up_bias) <{dtype = #ttcore.supportedDataTypes<bf16>}>
      : (tensor<2x128x1x5760xbf16>, tensor<1x1x1x5760xbf16>) -> tensor<2x128x1x5760xbf16>

    %gate = "ttnn.slice_static"(%pre) <{
      begins = [0 : i32, 0 : i32, 0 : i32, 1 : i32],
      ends = [2 : i32, 128 : i32, 1 : i32, 5760 : i32],
      step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]
    }> : (tensor<2x128x1x5760xbf16>) -> tensor<2x128x1x2880xbf16>
    %gate_silu = "ttnn.silu"(%gate)
      : (tensor<2x128x1x2880xbf16>) -> tensor<2x128x1x2880xbf16>

    %value = "ttnn.slice_static"(%pre) <{
      begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32],
      ends = [2 : i32, 128 : i32, 1 : i32, 5760 : i32],
      step = [1 : i32, 1 : i32, 1 : i32, 2 : i32]
    }> : (tensor<2x128x1x5760xbf16>) -> tensor<2x128x1x2880xbf16>
    %value_sigmoid = "ttnn.sigmoid"(%value)
      : (tensor<2x128x1x2880xbf16>) -> tensor<2x128x1x2880xbf16>

    %out = "ttnn.add"(%gate_silu, %value_sigmoid) <{dtype = #ttcore.supportedDataTypes<bf16>}>
      : (tensor<2x128x1x2880xbf16>, tensor<2x128x1x2880xbf16>) -> tensor<2x128x1x2880xbf16>
    %r1 = "ttnn.reshape"(%out) <{shape = [8 : i32, 32 : i32, 1 : i32, 2880 : i32]}>
      : (tensor<2x128x1x2880xbf16>) -> tensor<8x32x1x2880xbf16>
    %p1 = "ttnn.permute"(%r1) <{permutation = array<i64: 0, 2, 1, 3>}>
      : (tensor<8x32x1x2880xbf16>) -> tensor<8x1x32x2880xbf16>
    return %p1 : tensor<8x1x32x2880xbf16>
  }
}
