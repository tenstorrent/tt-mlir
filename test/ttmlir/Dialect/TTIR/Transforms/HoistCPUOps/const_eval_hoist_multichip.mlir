// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttcore-wrap-device-module --cpu-hoist-const-eval --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK: ttcore.device_module {
// CHECK: builtin.module

// =============================================================================
// Basic barrier segmentation
// =============================================================================

// --- Single all_gather barrier ---
// Compute chain before and after the CCL, each becoming a hoisted call.

// CHECK-LABEL: func.func private @single_all_gather_barrier
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @single_all_gather_barrier(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%add, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %sub = "ttir.subtract"(%gathered, %gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  %mul2 = "ttir.multiply"(%sub, %gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %mul2 : tensor<64x32xbf16>
}

// --- Only ops before CCL ---

// CHECK-LABEL: func.func private @ops_before_ccl_only
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK-NOT: call @cpu_hoisted
// CHECK: return
func.func private @ops_before_ccl_only(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%add, %add) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  return %gathered : tensor<64x32xbf16>
}

// --- Only ops after CCL ---

// CHECK-LABEL: func.func private @ops_after_ccl_only
// CHECK-NOT: call @cpu_hoisted
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @ops_after_ccl_only(
    %arg0: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %gathered = "ttir.all_gather"(%arg0) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %mul = "ttir.multiply"(%gathered, %gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  %sub = "ttir.subtract"(%mul, %gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %sub : tensor<64x32xbf16>
}

// --- CCL-only - nothing to hoist ---

// CHECK-LABEL: func.func private @ccl_only_no_hoist
// CHECK-NOT: call @cpu_hoisted
// CHECK: ttir.all_gather
// CHECK: return
func.func private @ccl_only_no_hoist(
    %arg0: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %gathered = "ttir.all_gather"(%arg0) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  return %gathered : tensor<64x32xbf16>
}

// =============================================================================
// Multiple barriers and consecutive CCLs
// =============================================================================

// --- Multiple barriers with multi-op segments ---

// CHECK-LABEL: func.func private @multiple_barriers_multi_op
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.add
// CHECK-NOT: ttir.multiply
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.subtract
// CHECK: ttir.all_reduce
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @multiple_barriers_multi_op(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%add, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %sub = "ttir.subtract"(%gathered, %gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  %reduced = "ttir.all_reduce"(%sub) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<64x32xbf16>) -> tensor<64x32xbf16>
  %mul2 = "ttir.multiply"(%reduced, %reduced) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  %add2 = "ttir.add"(%mul2, %reduced) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %add2 : tensor<64x32xbf16>
}

// --- Consecutive CCL barriers with no ops between ---
// Two back-to-back CCLs produce no middle segment.

// CHECK-LABEL: func.func private @consecutive_ccl_barriers
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: ttir.all_reduce
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @consecutive_ccl_barriers(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%add, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %reduced = "ttir.all_reduce"(%gathered) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<64x32xbf16>) -> tensor<64x32xbf16>
  %sub = "ttir.subtract"(%reduced, %reduced) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %sub : tensor<64x32xbf16>
}

// =============================================================================
// Different CCL barrier types
// =============================================================================

// --- All-reduce barrier ---

// CHECK-LABEL: func.func private @all_reduce_barrier
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_reduce
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @all_reduce_barrier(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%add, %add) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %reduced = "ttir.all_reduce"(%mul) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %sub = "ttir.subtract"(%reduced, %reduced) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %sub : tensor<32x32xbf16>
}

// --- Reduce-scatter barrier ---

// CHECK-LABEL: func.func private @reduce_scatter_barrier
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.reduce_scatter
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @reduce_scatter_barrier(
    %arg0: tensor<64x32xbf16>, %arg1: tensor<64x32xbf16>
) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  %rs = "ttir.reduce_scatter"(%add) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 0 : si32}> : (tensor<64x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%rs, %rs) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %mul : tensor<32x32xbf16>
}

// --- Non-identity MeshShard IS a barrier ---

// CHECK-LABEL: func.func private @devices_mesh_shard_is_barrier
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.mesh_shard
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @devices_mesh_shard_is_barrier(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<16x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %shard = "ttir.mesh_shard"(%add) <{shard_dims = array<i64: 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<32x32xbf16>) -> tensor<16x32xbf16>
  %mul = "ttir.multiply"(%shard, %shard) : (tensor<16x32xbf16>, tensor<16x32xbf16>) -> tensor<16x32xbf16>
  return %mul : tensor<16x32xbf16>
}

// =============================================================================
// MeshShard interactions
// =============================================================================

// --- Identity MeshShard is NOT a barrier ---
// Identity MeshShard ops are semantic no-ops. They must be included
// in the segment with their dependents, not treated as barriers.

// CHECK-LABEL: func.func private @identity_mesh_shard_not_barrier
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.add
// CHECK-NOT: ttir.mesh_shard
// CHECK-NOT: ttir.multiply
// CHECK: return
func.func private @identity_mesh_shard_not_barrier(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<32x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %shard = "ttir.mesh_shard"(%add) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%shard, %shard) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %mul : tensor<32x32xbf16>
}

// --- Identity MeshShard alongside a real barrier ---
// The identity MeshShard before the all_gather is a no-op and should
// be included in the first segment. The all_gather is the real barrier.

// CHECK-LABEL: func.func private @identity_mesh_shard_with_barrier
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.add
// CHECK-NOT: ttir.mesh_shard
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @identity_mesh_shard_with_barrier(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %shard = "ttir.mesh_shard"(%add) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%shard, %shard) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %sub = "ttir.subtract"(%gathered, %gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %sub : tensor<64x32xbf16>
}

// =============================================================================
// Inter-segment dependencies
// =============================================================================

// --- Segment 2 uses a function arg also used by segment 1 ---
// %arg1 feeds both the first segment and the second segment (via the
// function argument, not through the CCL). Both segments should be
// hoisted correctly with %arg1 as an input to both.

// CHECK-LABEL: func.func private @shared_func_arg_across_segments
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @shared_func_arg_across_segments(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%add) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  // %arg1 is 32x32, gathered is 64x32 — use reshape to match shapes.
  %reshaped = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1024 : i32]}> : (tensor<32x32xbf16>) -> tensor<1x1024xbf16>
  %reshaped_back = "ttir.reshape"(%reshaped) <{shape = [32 : i32, 32 : i32]}> : (tensor<1x1024xbf16>) -> tensor<32x32xbf16>
  %arg1_gathered = "ttir.all_gather"(%reshaped_back) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %mul = "ttir.multiply"(%gathered, %arg1_gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %mul : tensor<64x32xbf16>
}

// --- Multiple outputs from a segment feed the CCL and post-CCL ---
// Segment 1 produces two values: one feeds the CCL, the other is also
// consumed by segment 2 after the CCL via a function arg passthrough.

// CHECK-LABEL: func.func private @segment_multi_output
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.add
// CHECK-NOT: ttir.subtract
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @segment_multi_output(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %sub = "ttir.subtract"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered_add = "ttir.all_gather"(%add) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %gathered_sub = "ttir.all_gather"(%sub) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %mul = "ttir.multiply"(%gathered_add, %gathered_sub) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %mul : tensor<64x32xbf16>
}

// =============================================================================
// Skippable ops interacting with barriers
// =============================================================================

// --- Creation op consumed within a segment ---
// A zeros op used as an operand in the segment should be hoisted
// together with the segment (not skipped, since it's not at return).

// CHECK-LABEL: func.func private @creation_op_inside_segment
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.zeros
// CHECK-NOT: ttir.multiply
// CHECK: ttir.all_gather
// CHECK: return
func.func private @creation_op_inside_segment(
    %arg0: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%arg0, %zeros) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  return %gathered : tensor<64x32xbf16>
}

// --- Creation op at return with CCL in between ---
// The creation op chain at return is skipped. The segment before the
// CCL is hoisted. The CCL result is returned alongside the creation.

// CHECK-LABEL: func.func private @creation_at_return_with_ccl
// CHECK: ttir.zeros
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK-NOT: call @cpu_hoisted
// CHECK: return
func.func private @creation_at_return_with_ccl(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> (tensor<64x32xbf16>, tensor<64x32xbf16>) attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%add, %add) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %zeros = "ttir.zeros"() <{shape = array<i32: 64, 32>}> : () -> tensor<64x32xbf16>
  return %gathered, %zeros : tensor<64x32xbf16>, tensor<64x32xbf16>
}

// --- Creation op with transparent chain at return, plus CCL ---
// A zeros → reshape chain at return is skipped. The compute segment
// before the CCL is hoisted.

// CHECK-LABEL: func.func private @creation_chain_at_return_with_ccl
// CHECK: ttir.zeros
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: ttir.reshape
// CHECK-NOT: call @cpu_hoisted
// CHECK: return
func.func private @creation_chain_at_return_with_ccl(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> (tensor<64x32xbf16>, tensor<1x1024xbf16>) attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%add) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %zeros = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
  %reshaped = "ttir.reshape"(%zeros) <{shape = [1 : i32, 1024 : i32]}> : (tensor<32x32xbf16>) -> tensor<1x1024xbf16>
  return %gathered, %reshaped : tensor<64x32xbf16>, tensor<1x1024xbf16>
}

// =============================================================================
// Complex dependency patterns
// =============================================================================

// --- Diamond dependency through CCL ---
// Segment 1 produces a value that fans out to two CCLs, then segment 2
// combines both CCL results.

// CHECK-LABEL: func.func private @diamond_through_ccl
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: ttir.all_reduce
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @diamond_through_ccl(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%add, %add) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // Same value fans out to two different CCLs.
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %reduced = "ttir.all_reduce"(%mul) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // Combine results from both CCL paths.
  %reduced_gathered = "ttir.all_gather"(%reduced) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %result = "ttir.add"(%gathered, %reduced_gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %result : tensor<64x32xbf16>
}

// --- Long chain with interleaved CCLs ---
// A realistic pattern: compute → CCL → compute → CCL → compute → CCL → compute.

// CHECK-LABEL: func.func private @long_interleaved_chain
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_reduce
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @long_interleaved_chain(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<128x32xbf16> attributes {tt.function_type = "const_eval"} {
  // Segment 1
  %s1_add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %s1_mul = "ttir.multiply"(%s1_add, %s1_add) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // CCL 1
  %ccl1 = "ttir.all_gather"(%s1_mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  // Segment 2
  %s2_sub = "ttir.subtract"(%ccl1, %ccl1) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  %s2_add = "ttir.add"(%s2_sub, %ccl1) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  // CCL 2
  %ccl2 = "ttir.all_reduce"(%s2_add) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<64x32xbf16>) -> tensor<64x32xbf16>
  // Segment 3
  %s3_mul = "ttir.multiply"(%ccl2, %ccl2) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  // CCL 3
  %ccl3 = "ttir.all_gather"(%s3_mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<64x32xbf16>) -> tensor<128x32xbf16>
  // Segment 4
  %s4_sub = "ttir.subtract"(%ccl3, %ccl3) : (tensor<128x32xbf16>, tensor<128x32xbf16>) -> tensor<128x32xbf16>
  %s4_add = "ttir.add"(%s4_sub, %ccl3) : (tensor<128x32xbf16>, tensor<128x32xbf16>) -> tensor<128x32xbf16>
  return %s4_add : tensor<128x32xbf16>
}

// --- Segment produces multiple outputs consumed by different barriers ---
// Two values from segment 1 each feed a separate CCL, and the post-CCL
// segment consumes both CCL results.

// CHECK-LABEL: func.func private @multi_output_to_different_ccls
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK-NOT: ttir.add
// CHECK-NOT: ttir.subtract
// CHECK: ttir.all_gather
// CHECK: ttir.all_reduce
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: return
func.func private @multi_output_to_different_ccls(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  // Segment 1: produces %add and %sub.
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %sub = "ttir.subtract"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // %add → all_gather, %sub → all_reduce
  %gathered = "ttir.all_gather"(%add) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %reduced = "ttir.all_reduce"(%sub) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x32xbf16>) -> tensor<32x32xbf16>
  // Segment 2: consumes both CCL results.
  %reduced_gathered = "ttir.all_gather"(%reduced) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %mul = "ttir.multiply"(%gathered, %reduced_gathered) : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
  return %mul : tensor<64x32xbf16>
}

// =============================================================================
// Partial hoisting
// =============================================================================

// --- Non-lowerable op in one segment doesn't block others ---
// Only the segment with the non-lowerable rms_norm is skipped.

// CHECK-LABEL: func.func private @non_lowerable_in_one_segment
// CHECK: call @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttir.all_gather
// CHECK: ttir.rms_norm
// CHECK-NOT: call @cpu_hoisted
// CHECK: return
func.func private @non_lowerable_in_one_segment(
    %arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<32xbf16>
) -> tensor<64x32xbf16> attributes {tt.function_type = "const_eval"} {
  %add = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %mul = "ttir.multiply"(%add, %add) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %gathered = "ttir.all_gather"(%mul) <{all_gather_dim = 0 : si32, cluster_axis = 0 : ui32}> : (tensor<32x32xbf16>) -> tensor<64x32xbf16>
  %rms = "ttir.rms_norm"(%gathered, %arg2) <{normalized_shape = array<i64: 32>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<64x32xbf16>, tensor<32xbf16>) -> tensor<64x32xbf16>
  return %rms : tensor<64x32xbf16>
}

// Verify hoisted function declarations and definitions.
// CHECK: func.func private @cpu_hoisted_const_eval_{{.*}}
// CHECK: ttcore.cpu_module {
// CHECK: builtin.module {
// CHECK: func.func @cpu_hoisted_const_eval_{{.*}}
