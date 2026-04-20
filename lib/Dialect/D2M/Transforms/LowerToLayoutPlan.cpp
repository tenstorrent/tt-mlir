// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/LowerToLayoutPlan.h"

#include <optional>

namespace mlir::tt::d2m {

namespace {

// Return true iff the adjacent pair (a; b) reduces to nothing.
bool cancels(const Step &a, const Step &b) {
  // Tilize; Untilize → ∅ and Untilize; Tilize → ∅.
  if (std::holds_alternative<TilizeStep>(a) &&
      std::holds_alternative<UntilizeStep>(b)) {
    return true;
  }
  if (std::holds_alternative<UntilizeStep>(a) &&
      std::holds_alternative<TilizeStep>(b)) {
    return true;
  }
  return false;
}

// Return a fused Step iff the adjacent pair (a; b) merges into a single step.
std::optional<Step> tryFuse(const Step &a, const Step &b) {
  // F3: Reshard; Reshard → Reshard(second).
  if (std::holds_alternative<ReshardStep>(a) &&
      std::holds_alternative<ReshardStep>(b)) {
    return b;
  }
  // F6: Mask(v1); Mask(v2) → Mask(v2). Canonicalizer invariant: v2 != Undef.
  if (std::holds_alternative<MaskStep>(a) &&
      std::holds_alternative<MaskStep>(b)) {
    return b;
  }
  return std::nullopt;
}

bool applyCancellationPass(Plan &plan) {
  bool changed = false;
  for (size_t i = 0; i + 1 < plan.size();) {
    if (cancels(plan[i], plan[i + 1])) {
      plan.erase(plan.begin() + i, plan.begin() + i + 2);
      changed = true;
      // Re-check the new adjacency created by the erase.
      if (i > 0) {
        --i;
      }
    } else {
      ++i;
    }
  }
  return changed;
}

bool applyFusionPass(Plan &plan) {
  bool changed = false;
  for (size_t i = 0; i + 1 < plan.size();) {
    if (auto fused = tryFuse(plan[i], plan[i + 1])) {
      plan[i] = std::move(*fused);
      plan.erase(plan.begin() + i + 1);
      changed = true;
      // Re-check in case the fused step enables further fusion with its new
      // right neighbor.
    } else {
      ++i;
    }
  }
  return changed;
}

} // namespace

Plan canonicalize(const PlanState &source, const PlanState &target,
                  llvm::ArrayRef<int64_t> targetGridShape) {
  (void)source;
  (void)target;
  (void)targetGridShape;
  return {};
}

Plan minimize(Plan plan) {
  bool changed = true;
  while (changed) {
    changed = false;
    changed |= applyCancellationPass(plan);
    changed |= applyFusionPass(plan);
  }
  return plan;
}

} // namespace mlir::tt::d2m
