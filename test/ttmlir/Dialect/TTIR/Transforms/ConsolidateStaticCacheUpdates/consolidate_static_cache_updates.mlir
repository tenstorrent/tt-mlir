// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-consolidate-static-cache-updates %s | FileCheck %s

// Positive case: 3 add write-backs to the same static arg with equal deltas.
// Expect: 2 eliminated, 1 remains; all 3 return slots become the kept result.
// The kept result (last write-back) is broadcast to all consolidated slots so
// that every output address receives the correct incremented value.
module {
  // CHECK-LABEL: func.func @consolidate_three_to_one
  func.func @consolidate_three_to_one(
      %cumlen: tensor<1xi32>,
      %other: tensor<4x4xbf16>
  ) -> (tensor<1xi32>, tensor<4x4xbf16>, tensor<1xi32>, tensor<1xi32>) {
    %d0 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %d1 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %d2 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %a0 = "ttir.add"(%cumlen, %d0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a1 = "ttir.add"(%cumlen, %d1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a2 = "ttir.add"(%cumlen, %d2) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK:     [[ADD:%.+]] = "ttir.add"
    // CHECK-NOT: "ttir.add"
    // CHECK: return [[ADD]], %arg1, [[ADD]], [[ADD]]
    return %a0, %other, %a1, %a2 : tensor<1xi32>, tensor<4x4xbf16>, tensor<1xi32>, tensor<1xi32>
  }
}

// Negative case: different delta values — no consolidation.
module {
  // CHECK-LABEL: func.func @no_consolidate_different_deltas
  func.func @no_consolidate_different_deltas(
      %cumlen: tensor<1xi32>
  ) -> (tensor<1xi32>, tensor<1xi32>) {
    %d0 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %d1 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 2 : i32}> : () -> tensor<1xi32>
    %a0 = "ttir.add"(%cumlen, %d0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a1 = "ttir.add"(%cumlen, %d1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK: "ttir.add"
    // CHECK: "ttir.add"
    return %a0, %a1 : tensor<1xi32>, tensor<1xi32>
  }
}

// Negative case: add result has more than one use — no consolidation.
module {
  // CHECK-LABEL: func.func @no_consolidate_multi_use
  func.func @no_consolidate_multi_use(
      %cumlen: tensor<1xi32>,
      %other: tensor<1xi32>
  ) -> (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) {
    %d0 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %d1 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %a0 = "ttir.add"(%cumlen, %d0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a1 = "ttir.add"(%cumlen, %d1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // Both a0 slots consume %a0 — it has two uses, so it must not be erased.
    // CHECK: "ttir.add"
    // CHECK: "ttir.add"
    return %a0, %a1, %a0 : tensor<1xi32>, tensor<1xi32>, tensor<1xi32>
  }
}

// Positive case: TP-style per-layer pattern with different block args and
// ttir.constant delta.  Consolidation is safe because all per-layer
// cumulative_lengths always hold equal values at function entry (all layers
// advance together in each decode step), so returning one result to all output
// slots writes the correct updated value to every layer's buffer.
module {
  // CHECK-LABEL: func.func @consolidate_per_layer_constant
  func.func @consolidate_per_layer_constant(
      %cl0: tensor<1xi64>,
      %cl1: tensor<1xi64>,
      %cl2: tensor<1xi64>
  ) -> (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) {
    %delta = "ttir.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %a0 = "ttir.add"(%cl0, %delta) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %a1 = "ttir.add"(%cl1, %delta) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %a2 = "ttir.add"(%cl2, %delta) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    // CHECK:     [[ADD:%.+]] = "ttir.add"
    // CHECK-NOT: "ttir.add"
    // CHECK: return [[ADD]], [[ADD]], [[ADD]]
    return %a0, %a1, %a2 : tensor<1xi64>, tensor<1xi64>, tensor<1xi64>
  }
}

// Positive case: per-layer block args with internal uses — Phase 2 replaces all
// remaining uses of eliminated block args with the canonical arg so that
// internal consumers (e.g. update_cache position operands) collapse to one arg.
module {
  // CHECK-LABEL: func.func @consolidate_block_args_internal_uses
  func.func @consolidate_block_args_internal_uses(
      %cl0: tensor<1xi64>,
      %cl1: tensor<1xi64>,
      %cl2: tensor<1xi64>,
      %other: tensor<1xi64>
  ) -> (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) {
    %delta = "ttir.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %a0 = "ttir.add"(%cl0, %delta) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %a1 = "ttir.add"(%cl1, %delta) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    %a2 = "ttir.add"(%cl2, %delta) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    // Internal use of %cl0 — after Phase 2, %cl0 (%arg0) is replaced by
    // %cl2 (%arg2), the canonical arg, so this add becomes add(%arg2, %arg3).
    %internal = "ttir.add"(%cl0, %other) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    // Phase 1: only one ttir.add(blockArg, delta) survives.
    // CHECK:     [[KEPT:%.+]] = "ttir.add"(%arg2,
    // CHECK-NOT: "ttir.add"(%arg0, %delta
    // CHECK-NOT: "ttir.add"(%arg1, %delta
    // Phase 2: internal use of eliminated arg is replaced with canonical.
    // CHECK:     [[INT:%.+]] = "ttir.add"(%arg2, %arg3)
    // CHECK:     return [[KEPT]], [[KEPT]], [[KEPT]], [[INT]]
    return %a0, %a1, %a2, %internal : tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>
  }
}

// Negative case: BlockArgument type (tensor<1xi32>) differs from add result
// type (tensor<2xi32>) due to broadcasting — replacement would violate the
// function return type, so no consolidation.
module {
  // CHECK-LABEL: func.func @no_consolidate_type_mismatch
  func.func @no_consolidate_type_mismatch(
      %cumlen: tensor<1xi32>
  ) -> (tensor<2xi32>, tensor<2xi32>) {
    %d0 = "ttir.full"() <{shape = array<i32: 2>, fill_value = 1 : i32}> : () -> tensor<2xi32>
    %d1 = "ttir.full"() <{shape = array<i32: 2>, fill_value = 1 : i32}> : () -> tensor<2xi32>
    // broadcast: tensor<1xi32> + tensor<2xi32> -> tensor<2xi32>
    // blockArg type (tensor<1xi32>) != result type (tensor<2xi32>) — skip.
    %a0 = "ttir.add"(%cumlen, %d0) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
    %a1 = "ttir.add"(%cumlen, %d1) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
    // CHECK: "ttir.add"
    // CHECK: "ttir.add"
    return %a0, %a1 : tensor<2xi32>, tensor<2xi32>
  }
}
