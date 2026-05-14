// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-consolidate-static-cache-updates --cse %s | FileCheck %s

// Helper: each test func that wants its scalar args treated as
// cumulative_length anchors them through a `ttir.update_cache` op (chained
// when there are multiple args), since the pass infers cumulative_length
// args by tracing back from update-style cache ops in the same function.

// Positive case: 3 add write-backs to the same static arg with equal deltas.
// The pass does nothing here (single block arg, no arg unification needed).
// CSE deduplicates the 3 identical ttir.full + ttir.add ops down to 1.
module {
  // CHECK-LABEL: func.func @consolidate_three_to_one
  func.func @consolidate_three_to_one(
      %cumlen: tensor<1xi32>,
      %other: tensor<4x4xbf16>,
      %cache: tensor<1x8x16x128xbf16>,
      %input: tensor<1x8x1x128xbf16>
  ) -> (tensor<1xi32>, tensor<4x4xbf16>, tensor<1xi32>, tensor<1xi32>, tensor<1x8x16x128xbf16>) {
    %d0 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %d1 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %d2 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %a0 = "ttir.add"(%cumlen, %d0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a1 = "ttir.add"(%cumlen, %d1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a2 = "ttir.add"(%cumlen, %d2) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // Anchor %cumlen as cumulative_length.
    %c0 = "ttir.update_cache"(%cache, %input, %cumlen) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    // CHECK:     [[ADD:%.+]] = "ttir.add"
    // CHECK-NOT: "ttir.add"
    // CHECK: return [[ADD]], %arg1, [[ADD]], [[ADD]]
    return %a0, %other, %a1, %a2, %c0 : tensor<1xi32>, tensor<4x4xbf16>, tensor<1xi32>, tensor<1xi32>, tensor<1x8x16x128xbf16>
  }
}

// Negative case: different delta values — no consolidation.
module {
  // CHECK-LABEL: func.func @no_consolidate_different_deltas
  func.func @no_consolidate_different_deltas(
      %cumlen: tensor<1xi32>,
      %cache: tensor<1x8x16x128xbf16>,
      %input: tensor<1x8x1x128xbf16>
  ) -> (tensor<1xi32>, tensor<1xi32>, tensor<1x8x16x128xbf16>) {
    %d0 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %d1 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 2 : i32}> : () -> tensor<1xi32>
    %a0 = "ttir.add"(%cumlen, %d0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a1 = "ttir.add"(%cumlen, %d1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %c0 = "ttir.update_cache"(%cache, %input, %cumlen) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    // CHECK: "ttir.add"
    // CHECK: "ttir.add"
    return %a0, %a1, %c0 : tensor<1xi32>, tensor<1xi32>, tensor<1x8x16x128xbf16>
  }
}

// Positive case: TP-style per-layer pattern with different block args.
// The pass unifies %arg0 and %arg1 → %arg2 (canonical).
// CSE then deduplicates the 3 identical adds to 1.  Safe because all per-layer
// cumulative_lengths hold equal values at function entry (lockstep invariant).
module {
  // CHECK-LABEL: func.func @consolidate_per_layer
  func.func @consolidate_per_layer(
      %arg0: tensor<1xi32>,
      %arg1: tensor<1xi32>,
      %arg2: tensor<1xi32>,
      %cache: tensor<1x8x16x128xbf16>,
      %input: tensor<1x8x1x128xbf16>
  ) -> (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x8x16x128xbf16>) {
    %delta = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %a0 = "ttir.add"(%arg0, %delta) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a1 = "ttir.add"(%arg1, %delta) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a2 = "ttir.add"(%arg2, %delta) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // Anchor each arg as cumulative_length via chained cache updates.
    %c0 = "ttir.update_cache"(%cache, %input, %arg0) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    %c1 = "ttir.update_cache"(%c0, %input, %arg1) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    %c2 = "ttir.update_cache"(%c1, %input, %arg2) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    // CHECK:     [[ADD:%.+]] = "ttir.add"
    // CHECK-NOT: "ttir.add"
    // CHECK: return [[ADD]], [[ADD]], [[ADD]]
    return %a0, %a1, %a2, %c2 : tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x8x16x128xbf16>
  }
}

// Positive case: per-layer block args with internal uses.  The pass replaces
// all uses of %arg0 and %arg1 with canonical %arg2.  CSE collapses the 3
// identical write-back adds to 1.  The internal add uses %arg2 after unification.
// %arg3 is NOT a cumulative_length (no update_cache anchors it), so it is
// untouched by the pass.
module {
  // CHECK-LABEL: func.func @consolidate_block_args_internal_uses
  func.func @consolidate_block_args_internal_uses(
      %arg0: tensor<1xi32>,
      %arg1: tensor<1xi32>,
      %arg2: tensor<1xi32>,
      %arg3: tensor<1xi32>,
      %cache: tensor<1x8x16x128xbf16>,
      %input: tensor<1x8x1x128xbf16>
  ) -> (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x8x16x128xbf16>) {
    %delta = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %a0 = "ttir.add"(%arg0, %delta) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a1 = "ttir.add"(%arg1, %delta) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a2 = "ttir.add"(%arg2, %delta) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // Internal use of %arg0 — the pass replaces %arg0 with the canonical
    // %arg2, so this add becomes add(%arg2, %arg3).
    %internal = "ttir.add"(%arg0, %arg3) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // Anchor %arg0/%arg1/%arg2 as cumulative_length. %arg3 is intentionally
    // NOT consumed by any cache op so it does not get classified.
    %c0 = "ttir.update_cache"(%cache, %input, %arg0) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    %c1 = "ttir.update_cache"(%c0, %input, %arg1) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    %c2 = "ttir.update_cache"(%c1, %input, %arg2) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    // After arg unification + CSE: one write-back add survives, %arg0/%arg1 gone.
    // CHECK:     [[KEPT:%.+]] = "ttir.add"(%arg2,
    // CHECK-NOT: "ttir.add"(%arg0,
    // CHECK-NOT: "ttir.add"(%arg1,
    // CHECK:     [[INT:%.+]] = "ttir.add"(%arg2, %arg3)
    // CHECK:     return [[KEPT]], [[KEPT]], [[KEPT]], [[INT]]
    return %a0, %a1, %a2, %internal, %c2 : tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1x8x16x128xbf16>
  }
}

// Negative case: BlockArgument type (tensor<1xi32>) differs from add result
// type (tensor<2xi32>) due to broadcasting — block arg replacement is skipped.
// Different delta values prevent CSE from merging, keeping both adds in output.
module {
  // CHECK-LABEL: func.func @no_consolidate_type_mismatch
  func.func @no_consolidate_type_mismatch(
      %cumlen: tensor<1xi32>,
      %cache: tensor<1x8x16x128xbf16>,
      %input: tensor<1x8x1x128xbf16>
  ) -> (tensor<2xi32>, tensor<2xi32>, tensor<1x8x16x128xbf16>) {
    %d0 = "ttir.full"() <{shape = array<i32: 2>, fill_value = 1 : i32}> : () -> tensor<2xi32>
    %d1 = "ttir.full"() <{shape = array<i32: 2>, fill_value = 2 : i32}> : () -> tensor<2xi32>
    // broadcast: tensor<1xi32> + tensor<2xi32> -> tensor<2xi32>
    // blockArg type (tensor<1xi32>) != result type (tensor<2xi32>) — skip.
    %a0 = "ttir.add"(%cumlen, %d0) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
    %a1 = "ttir.add"(%cumlen, %d1) : (tensor<1xi32>, tensor<2xi32>) -> tensor<2xi32>
    %c0 = "ttir.update_cache"(%cache, %input, %cumlen) <{batch_offset = 0 : i32}>
        : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    // CHECK: "ttir.add"
    // CHECK: "ttir.add"
    return %a0, %a1, %c0 : tensor<2xi32>, tensor<2xi32>, tensor<1x8x16x128xbf16>
  }
}

// Negative case: no `ttir.update_cache` op consumes the args, so the
// inference inside the pass finds no cumulative_length args and the pass is a
// no-op. Two independent counters with the same delta are NOT unified.
module {
  // CHECK-LABEL: func.func @no_consolidate_no_cache_op
  func.func @no_consolidate_no_cache_op(
      %arg0: tensor<1xi32>,
      %arg1: tensor<1xi32>
  ) -> (tensor<1xi32>, tensor<1xi32>) {
    %delta = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1 : i32}> : () -> tensor<1xi32>
    %a0 = "ttir.add"(%arg0, %delta) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %a1 = "ttir.add"(%arg1, %delta) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK:     [[A0:%.+]] = "ttir.add"(%arg0,
    // CHECK:     [[A1:%.+]] = "ttir.add"(%arg1,
    // CHECK:     return [[A0]], [[A1]]
    return %a0, %a1 : tensor<1xi32>, tensor<1xi32>
  }
}
