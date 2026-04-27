// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-consolidate-static-cache-updates --cse %s | FileCheck %s

// Positive case: 3 add write-backs to the same static arg with equal deltas.
// The pass does nothing here (single block arg, no arg unification needed).
// CSE deduplicates the 3 identical ttir.full + ttir.add ops down to 1.
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

// Positive case: TP-style per-layer pattern with different block args and
// ttir.constant delta.  The pass unifies %cl0 and %cl1 → %cl2 (canonical).
// CSE then deduplicates the 3 identical adds to 1.  Safe because all per-layer
// cumulative_lengths hold equal values at function entry (lockstep invariant).
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

// Positive case: per-layer block args with internal uses.  The pass replaces
// all uses of %cl0 and %cl1 with canonical %cl2.  CSE collapses the 3
// identical write-back adds to 1.  The internal add uses %cl2 after unification.
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
    // Internal use of %cl0 — the pass replaces %cl0 (%arg0) with the
    // canonical %cl2 (%arg2), so this add becomes add(%arg2, %arg3).
    %internal = "ttir.add"(%cl0, %other) : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    // After arg unification + CSE: one write-back add survives, %arg0/%arg1 gone.
    // CHECK:     [[KEPT:%.+]] = "ttir.add"(%arg2,
    // CHECK-NOT: "ttir.add"(%arg0,
    // CHECK-NOT: "ttir.add"(%arg1,
    // CHECK:     [[INT:%.+]] = "ttir.add"(%arg2, %arg3)
    // CHECK:     return [[KEPT]], [[KEPT]], [[KEPT]], [[INT]]
    return %a0, %a1, %a2, %internal : tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>
  }
}

// Negative case: BlockArgument type (tensor<1xi32>) differs from add result
// type (tensor<2xi32>) due to broadcasting — block arg replacement is skipped.
// Different delta values prevent CSE from merging, keeping both adds in output.
module {
  // CHECK-LABEL: func.func @no_consolidate_type_mismatch
  func.func @no_consolidate_type_mismatch(
      %cumlen: tensor<1xi32>
  ) -> (tensor<2xi32>, tensor<2xi32>) {
    %d0 = "ttir.full"() <{shape = array<i32: 2>, fill_value = 1 : i32}> : () -> tensor<2xi32>
    %d1 = "ttir.full"() <{shape = array<i32: 2>, fill_value = 2 : i32}> : () -> tensor<2xi32>
    // broadcast: tensor<1xi32> + tensor<2xi32> -> tensor<2xi32>
    // blockArg type (tensor<1xi32>) != result type (tensor<2xi32>) — skip.
    %a0 = "ttir.add"(%cumlen, %d0) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
    %a1 = "ttir.add"(%cumlen, %d1) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
    // CHECK: "ttir.add"
    // CHECK: "ttir.add"
    return %a0, %a1 : tensor<2xi32>, tensor<2xi32>
  }
}
