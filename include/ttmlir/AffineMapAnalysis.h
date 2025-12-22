// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_AFFINEMAPANALYSIS_H
#define TTMLIR_AFFINEMAPANALYSIS_H

#include "ttmlir/Asserts.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <numeric>
#include <optional>

namespace ttmlir::utils {

inline std::optional<mlir::AffineDimExpr> isDimExpr(mlir::AffineExpr expr) {
  auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr);
  return dimExpr ? std::optional<mlir::AffineDimExpr>(dimExpr) : std::nullopt;
}

inline std::optional<int64_t> getSumOfModuli(mlir::AffineExpr expr) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    if (binOp.getKind() == mlir::AffineExprKind::Add) {
      auto lhs = getSumOfModuli(binOp.getLHS());
      auto rhs = getSumOfModuli(binOp.getRHS());
      if (lhs.has_value() && rhs.has_value()) {
        return (lhs.value() + rhs.value());
      }
    } else if (binOp.getKind() == mlir::AffineExprKind::Mod) {
      if (auto rhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getRHS())) {
        // We need to subtract 1, as the max value of the modulo op is one less
        // than the modulus value.
        return (rhsConst.getValue() - 1);
      }
    } else if (binOp.getKind() == mlir::AffineExprKind::Mul) {
      if (auto rhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getRHS())) {
        auto lhs = getSumOfModuli(binOp.getLHS());
        if (lhs.has_value()) {
          return lhs.value() * rhsConst.getValue();
        }
      }
      if (auto lhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getLHS())) {
        auto rhs = getSumOfModuli(binOp.getRHS());
        if (rhs.has_value()) {
          return rhs.value() * lhsConst.getValue();
        }
      }
    }
  }
  return std::nullopt;
}

inline mlir::AffineExpr simplifyZeroFloorDivExpr(mlir::AffineExpr expr) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = simplifyZeroFloorDivExpr(binOp.getLHS());
    auto rhs = simplifyZeroFloorDivExpr(binOp.getRHS());

    if (binOp.getKind() == mlir::AffineExprKind::FloorDiv) {
      if (auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
        int64_t divisor = rhsConst.getValue();
        auto modSum = getSumOfModuli(lhs);
        if (modSum.has_value() && modSum.value() < divisor) {
          return mlir::getAffineConstantExpr(0, expr.getContext());
        }
      }
    }

    switch (binOp.getKind()) {
    case mlir::AffineExprKind::Add:
      return lhs + rhs;
    case mlir::AffineExprKind::Mul:
      return lhs * rhs;
    case mlir::AffineExprKind::Mod:
      return lhs % rhs;
    case mlir::AffineExprKind::FloorDiv:
      return lhs.floorDiv(rhs);
    case mlir::AffineExprKind::CeilDiv:
      return lhs.ceilDiv(rhs);
    default:
      return expr;
    }
  }
  return expr;
}

/// Simplifies the affine map by finding sub expressions in results that always
/// evaluate to zero.
/// Specifically, it looks for: ((dim0 mod M) + (dim1 mod N) ... ) floorDiv Q.
/// If sum(M, N, ...) <= Q, the expression is replaced with 0.
inline mlir::AffineMap simplifyZeroFloorDiv(mlir::AffineMap map) {
  mlir::SmallVector<mlir::AffineExpr> newResults;
  for (auto result : map.getResults()) {
    newResults.push_back(simplifyZeroFloorDivExpr(result));
  }
  return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), newResults,
                              map.getContext());
}

inline std::optional<int64_t>
getExprUpperBound(mlir::AffineExpr expr, mlir::ArrayRef<int64_t> dimBounds) {

  if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (dimExpr.getPosition() < dimBounds.size()) {
      return dimBounds[dimExpr.getPosition()] - 1;
    }
    return std::nullopt;
  }
  if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    // We conservatively return nullopt for negative constants to avoid
    // handling sign issues in Mul/Div for upper bound calculation.
    if (constExpr.getValue() < 0) {
      return std::nullopt;
    }
    return constExpr.getValue();
  }
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {

    auto lhs = getExprUpperBound(binOp.getLHS(), dimBounds);

    // Quick check for Mod with constant RHS.
    if (binOp.getKind() == mlir::AffineExprKind::Mod) {

      auto lhs = getExprUpperBound(binOp.getLHS(), dimBounds);
      if (auto rhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getRHS())) {
        int64_t rhsVal = rhsConst.getValue();
        if (rhsVal <= 0) {
          return std::nullopt;
        }
        if (lhs.has_value()) {
          auto r = std::min(lhs.value(), rhsVal - 1);
          return r;
        }
        return rhsVal - 1;
      }
      return std::nullopt;
    }

    if (binOp.getKind() == mlir::AffineExprKind::FloorDiv) {
      if (auto rhsConst =
              llvm::dyn_cast<mlir::AffineConstantExpr>(binOp.getRHS())) {
        int64_t rhsVal = rhsConst.getValue();
        if (rhsVal <= 0) {
          return std::nullopt;
        }
        if (lhs.has_value()) {
          return lhs.value() / rhsVal;
        }
      }
      return std::nullopt;
    }

    auto rhs = getExprUpperBound(binOp.getRHS(), dimBounds);

    if (lhs.has_value() && rhs.has_value()) {
      switch (binOp.getKind()) {
      case mlir::AffineExprKind::Add: {
        return lhs.value() + rhs.value();
      }
      case mlir::AffineExprKind::Mul: {
        return lhs.value() * rhs.value();
      }
      default:
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

inline mlir::AffineExpr
simplifyAffineExprWithRangeAnalysis(mlir::AffineExpr expr,
                                    mlir::ArrayRef<int64_t> dimBounds) {
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    auto lhs = simplifyAffineExprWithRangeAnalysis(binOp.getLHS(), dimBounds);
    auto rhs = simplifyAffineExprWithRangeAnalysis(binOp.getRHS(), dimBounds);

    if (binOp.getKind() == mlir::AffineExprKind::Mod) {
      if (auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
        auto lhsUB = getExprUpperBound(lhs, dimBounds);
        if (lhsUB.has_value() && lhsUB.value() < rhsConst.getValue()) {
          return lhs;
        }
      }
      return lhs % rhs;
    }
    if (binOp.getKind() == mlir::AffineExprKind::FloorDiv) {
      if (auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs)) {
        auto lhsUB = getExprUpperBound(lhs, dimBounds);
        if (lhsUB.has_value() && lhsUB.value() < rhsConst.getValue()) {
          return mlir::getAffineConstantExpr(0, expr.getContext());
        }
      }
      return lhs.floorDiv(rhs);
    }

    switch (binOp.getKind()) {
    case mlir::AffineExprKind::Add:
      return lhs + rhs;
    case mlir::AffineExprKind::Mul:
      return lhs * rhs;
    case mlir::AffineExprKind::FloorDiv:
      return lhs.floorDiv(rhs);
    case mlir::AffineExprKind::CeilDiv:
      return lhs.ceilDiv(rhs);
    default:
      return expr;
    }
  }
  return expr;
}

/// Simplifies an affine map by eliminating redundant mod/floordiv operations
/// using range analysis. If the upper bound of an operand is less than the
/// modulus, the mod is removed; if less than the divisor, floordiv becomes 0.
/// When performEquivalenceCheck is true, samples the domain to verify the
/// simplified map is equivalent; returns the original map if not.
inline mlir::AffineMap
simplifyAffineMapWithRangeAnalysis(mlir::AffineMap map,
                                   mlir::ArrayRef<int64_t> dimBounds,
                                   bool performEquivalenceCheck = false) {
  TT_assertv(map.getNumDims() == dimBounds.size(),
             "Number of dimension bounds must match number of map dimensions");
  mlir::SmallVector<mlir::AffineExpr> newResults;
  for (auto result : map.getResults()) {
    newResults.push_back(
        simplifyAffineExprWithRangeAnalysis(result, dimBounds));
  }
  auto simplifiedMap = mlir::AffineMap::get(
      map.getNumDims(), map.getNumSymbols(), newResults, map.getContext());

  // Roundtrip equivalence check: sample both maps over the entire domain
  // and verify they produce identical results at every point.
  if (performEquivalenceCheck) {
    bool equivalent = true;
    sample(dimBounds, [&](llvm::ArrayRef<int64_t> point) {
      if (map.compose(point) != simplifiedMap.compose(point)) {
        equivalent = false;
      }
    });
    if (!equivalent) {
      return map;
    }
  }

  return simplifiedMap;
}

inline mlir::AffineExpr isBinaryAdd(mlir::AffineExpr expr) {
  auto lhsBinExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  return (lhsBinExpr && lhsBinExpr.getKind() == mlir::AffineExprKind::Add)
             ? lhsBinExpr
             : mlir::AffineExpr{};
}

inline std::optional<mlir::AffineBinaryOpExpr>
isBinaryMul(mlir::AffineExpr expr) {
  auto lhsBinExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  return (lhsBinExpr && lhsBinExpr.getKind() == mlir::AffineExprKind::Mul)
             ? std::optional<mlir::AffineBinaryOpExpr>(lhsBinExpr)
             : std::nullopt;
}

inline void
collectSumOperandsImpl(mlir::AffineExpr expr,
                       llvm::SmallVectorImpl<mlir::AffineExpr> &results) {
  // Add operations have least precedence, so collecting all operands
  // commutatively gathers the entire expressions; everything else must be a
  // child expr of the top-level add.
  if (auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr)) {
    if (binOp.getKind() == mlir::AffineExprKind::Add) {
      collectSumOperandsImpl(binOp.getLHS(), results);
      collectSumOperandsImpl(binOp.getRHS(), results);
      return;
    }
  }
  results.push_back(expr);
}

inline mlir::SmallVector<mlir::AffineExpr>
collectSumOperands(mlir::AffineExpr expr) {
  mlir::SmallVector<mlir::AffineExpr> ops;
  collectSumOperandsImpl(expr, ops);
  return ops;
}

/// Finds the minimum gap to the next target boundary for any achievable sum.
/// For floor division semantics: gap = target - (sum % target), or target if
/// sum is exactly on a boundary.
/// Returns the minimum such gap across all achievable sums.
inline int64_t minimizeGap(int64_t target, llvm::ArrayRef<int64_t> multipliers,
                           llvm::ArrayRef<int64_t> bounds,
                           int64_t constOffset) {
  assert(multipliers.size() == bounds.size());

  // Compute all achievable remainders (mod target) using dynamic programming.
  // Since we only care about sum % target, we track remainders directly.
  // This limits the set size to at most 'target' elements, preventing
  // exponential blowup.
  llvm::DenseSet<int64_t> achievableRemainders;
  achievableRemainders.insert(((constOffset % target) + target) % target);

  for (size_t i = 0; i < multipliers.size(); ++i) {
    llvm::DenseSet<int64_t> newRemainders;
    int64_t multiplierMod = ((multipliers[i] % target) + target) % target;
    for (int64_t remainder : achievableRemainders) {
      for (int64_t n = 0; n <= bounds[i]; ++n) {
        int64_t newRemainder = (remainder + multiplierMod * n) % target;
        newRemainders.insert(newRemainder);
        // Early exit: if we've covered all remainders, no need to continue
        if (newRemainders.size() == static_cast<size_t>(target)) {
          break;
        }
      }
      if (newRemainders.size() == static_cast<size_t>(target)) {
        break;
      }
    }
    achievableRemainders = std::move(newRemainders);
    // Early exit: if all remainders are achievable, gap will be 1
    if (achievableRemainders.size() == static_cast<size_t>(target)) {
      return 1;
    }
  }

  // Find the minimum gap to the next boundary for any achievable remainder.
  // For floor division: gap = target - remainder, or target if remainder is 0.
  int64_t bestGap = target;

  for (int64_t remainder : achievableRemainders) {
    int64_t gap = (remainder == 0) ? target : (target - remainder);
    if (gap < bestGap) {
      bestGap = gap;
    }
    // Early exit if gap is 1 (best possible non-exact alignment)
    if (bestGap == 1) {
      break;
    }
  }

  return bestGap;
}

inline bool exprContainsDim(mlir::AffineExpr expr, unsigned dimPos) {
  bool foundDim = false;
  expr.walk([&](mlir::AffineExpr e) {
    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(e)) {
      if (dimExpr.getPosition() == dimPos) {
        foundDim = true;
      }
    }
  });
  return foundDim;
}

inline bool exprIsSpecificDimExpr(mlir::AffineExpr expr, unsigned dimPos) {
  return llvm::isa<mlir::AffineDimExpr>(expr) &&
         llvm::dyn_cast<mlir::AffineDimExpr>(expr).getPosition() == dimPos;
}

/// Recursively analyzes contiguity of an affine expression to find the maximum
/// coalescable run. Returns -1 if unconstrained, 1 on error, or the contiguity
/// bound.
inline int64_t analyzeShardResultExprForContiguity(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &dimBounds,
    unsigned dimPos, unsigned numGridDims = 0,
    std::optional<int64_t> parentModulus = std::nullopt) {
  // Helper to check if steps exceed dim bounds.
  auto checkBoundsAndReturn = [&](int64_t stepsNeeded) -> int64_t {
    int64_t targetDimBound = dimBounds.lookup(dimPos);
    if (stepsNeeded >= targetDimBound) {
      return -1; // Unconstrained - always contiguous within bounds.
    }
    return stepsNeeded;
  };
  // Handle constant expressions.
  if (auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr)) {
    if (constExpr.getValue() < 0) {
      return 1; // Error: negative constant.
    }
    return -1; // No contiguity constraint.
  }

  // Handle dim expressions.
  if (llvm::isa<mlir::AffineDimExpr>(expr)) {
    return -1; // No contiguity constraint.
  }

  // Handle binary operations.
  auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!binOp) {
    return 1;
  }

  mlir::AffineExpr lhs = binOp.getLHS();
  mlir::AffineExpr rhs = binOp.getRHS();

  auto kind = binOp.getKind();
  switch (kind) {
  case mlir::AffineExprKind::Add: {
    // Analyze each operand recursively, propagating rhsConst.
    int64_t lhsBound = analyzeShardResultExprForContiguity(
        lhs, dimBounds, dimPos, numGridDims, parentModulus);
    int64_t rhsBound = analyzeShardResultExprForContiguity(
        rhs, dimBounds, dimPos, numGridDims, parentModulus);

    // Return GCD of bounds; -1 means unconstrained.
    if (lhsBound == -1) {
      return rhsBound;
    }
    if (rhsBound == -1) {
      return lhsBound;
    }
    return std::gcd(lhsBound, rhsBound);
  }

  case mlir::AffineExprKind::Mul: {
    // Mul must have a constant operand.
    auto lhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(lhs);
    auto rhsConstExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);

    if (!lhsConst && !rhsConstExpr) {
      return 1; // Error: mul without constant operand.
    }

    int64_t constVal =
        rhsConstExpr ? rhsConstExpr.getValue() : lhsConst.getValue();

    if (constVal < 0) {
      return 1; // Error: negative constant.
    }

    int64_t lhsBound = analyzeShardResultExprForContiguity(
        lhs, dimBounds, dimPos, numGridDims, parentModulus);

    // If lhs is constrained, propagate the constraint up.
    if (lhsBound != -1) {
      return lhsBound;
    }

    // If rhsConst is not set, the contiguity bound is unconstrained.
    if (!parentModulus.has_value()) {
      return -1;
    }

    // If rhsConst % constVal == 0, bound is rhsConst / constVal.
    if (*parentModulus % constVal == 0) {
      return *parentModulus / constVal;
    }
    // If rhsConst % constVal != 0, return error.
    return 1;
  }

  case mlir::AffineExprKind::Mod:
  case mlir::AffineExprKind::FloorDiv:
  case mlir::AffineExprKind::CeilDiv: {
    // RHS must be constant.
    auto rhsConstExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    assert(rhsConstExpr && "Mod/FloorDiv RHS must be constant expression");

    int64_t modulus = rhsConstExpr.getValue();
    if (modulus <= 0) {
      return 1;
    }

    // Check if LHS is also floordiv/mod - error condition.
    if (auto lhsBinOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(lhs)) {
      auto lhsKind = lhsBinOp.getKind();
      if (lhsKind == mlir::AffineExprKind::Mod ||
          lhsKind == mlir::AffineExprKind::FloorDiv) {
        if (!exprContainsDim(lhsBinOp.getLHS(), dimPos)) {
          return -1;
        }
        // Precompute the value of the RHS constant for lhsBinOp.
        auto innerRhsConst =
            llvm::dyn_cast<mlir::AffineConstantExpr>(lhsBinOp.getRHS());
        TT_assertv(innerRhsConst, "Nested binary op RHS must be constant");
        int64_t innerRhsConstValue = innerRhsConst.getValue();

        if (lhsKind == mlir::AffineExprKind::FloorDiv &&
            kind == mlir::AffineExprKind::FloorDiv) {
          // Both inner and outer are floordiv, so collapse them into a
          // single floordiv.
          int64_t combinedDivisor = innerRhsConstValue * modulus;
          return analyzeShardResultExprForContiguity(
              lhsBinOp.getLHS().floorDiv(combinedDivisor), dimBounds, dimPos,
              numGridDims, combinedDivisor);
        }
        if (lhsKind == mlir::AffineExprKind::Mod &&
            kind == mlir::AffineExprKind::Mod) {
          // Both inner and outer are mod: recurse on the LHS with min of the
          // two rhs modulus.
          int64_t gcdMod = std::min(innerRhsConstValue, modulus);
          return analyzeShardResultExprForContiguity(lhsBinOp.getLHS() % gcdMod,
                                                     dimBounds, dimPos,
                                                     numGridDims, gcdMod);
        }
        if (lhsKind == mlir::AffineExprKind::Mod &&
            kind == mlir::AffineExprKind::FloorDiv) {
          // Inner is mod, outer is floordiv: analyze mod op, then multiply by
          // floordiv divisor.
          int64_t modBound = analyzeShardResultExprForContiguity(
              lhsBinOp, dimBounds, dimPos, numGridDims, innerRhsConstValue);
          return modBound * modulus;
        }
        if (lhsKind == mlir::AffineExprKind::FloorDiv &&
            kind == mlir::AffineExprKind::Mod) {
          // Discontinuity happens as soon as lhs floordiv expr changes; no
          // need to analyze outer mod.
          return analyzeShardResultExprForContiguity(
              lhsBinOp, dimBounds, dimPos, numGridDims, modulus);
        }
        // Not handled.
        return -1;
      }
    }

    // If LHS is plain dim expr, check if it's the target dim.
    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(lhs)) {
      if (dimExpr.getPosition() == dimPos) {
        return checkBoundsAndReturn(modulus);
      }
      // Different dim - unconstrained for target dim.
      return -1;
    }

    // If LHS is a Mul, return result from calling func on lhs with boundValue.
    if (auto lhsBinOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(lhs)) {
      if (lhsBinOp.getKind() == mlir::AffineExprKind::Mul) {
        return analyzeShardResultExprForContiguity(lhs, dimBounds, dimPos,
                                                   numGridDims, modulus);
      }

      // If LHS is an Add, find worst-case gap using minimizeGap algorithm.
      // This handles patterns like (A*d0 + B*d1 + ... + C) mod N.
      if (lhsBinOp.getKind() == mlir::AffineExprKind::Add) {
        auto sumOperands = collectSumOperands(lhs);

        // Track target dim separately from other dims.
        int64_t targetMultiplier = 0;
        bool foundTargetDim = false;
        // Track (dimPosition, multiplier) pairs for other dimensions
        llvm::SmallVector<std::pair<unsigned, int64_t>> otherDimInfo;
        int64_t constSum = 0;

        for (auto sumOperand : sumOperands) {
          // Case 0: constants - accumulate into constSum.
          if (auto constExpr =
                  llvm::dyn_cast<mlir::AffineConstantExpr>(sumOperand)) {
            constSum += constExpr.getValue();
            continue;
          }

          // Case 1: Plain dim expression (multiplier = 1).
          if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(sumOperand)) {
            if (dimExpr.getPosition() == dimPos) {
              if (foundTargetDim) {
                return 1; // Multiple occurrences of target dim, fallback.
              }
              foundTargetDim = true;
              targetMultiplier = 1;
            } else {
              otherDimInfo.push_back({dimExpr.getPosition(), 1});
            }
            continue;
          }

          // Case 2: Mul expression (dim * const).
          if (auto mulOp = isBinaryMul(sumOperand)) {
            auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(mulOp->getLHS());
            auto constExpr =
                llvm::dyn_cast<mlir::AffineConstantExpr>(mulOp->getRHS());

            if (!dimExpr || !constExpr) {
              return 1; // Cannot analyze, fallback.
            }

            int64_t mulValue = constExpr.getValue();

            if (dimExpr.getPosition() == dimPos) {
              if (foundTargetDim) {
                return 1; // Multiple occurrences of target dim, fallback.
              }
              foundTargetDim = true;
              targetMultiplier = mulValue;
            } else {
              otherDimInfo.push_back({dimExpr.getPosition(), mulValue});
            }
            continue;
          }

          // Cannot analyze other patterns.
          return 1;
        }

        // If we found the target dim, compute worst-case contiguity.
        if (foundTargetDim && targetMultiplier > 0) {
          if (constSum < 0) {
            return 1; // Negative constants not analyzed, fallback.
          }

          int64_t gap;
          if (otherDimInfo.empty()) {
            // Simple case: just the target dim and a constant offset.
            // Gap is distance from constSum to the next modulus boundary.
            int64_t remainder = constSum % modulus;
            gap = (remainder == 0) ? modulus : (modulus - remainder);
          } else {
            // Build multipliers and bounds arrays from tracked dimension info.
            // Include ALL other dimensions to find worst-case gap:
            //   - Grid dims: fixed per transfer, need worst case across grid
            //   - Outer shard dims: fixed during inner iteration, need worst
            //   case
            //   - Inner shard dims: vary within iteration
            // All must be considered to find the minimum gap that produces
            // the worst-case contiguity bound.
            llvm::SmallVector<int64_t> otherMultipliers;
            llvm::SmallVector<int64_t> otherBounds;
            for (auto [dimPosition, mul] : otherDimInfo) {
              otherMultipliers.push_back(mul);
              // Bound is (dimSize - 1) since dim ranges from 0 to dimSize-1
              otherBounds.push_back(dimBounds.lookup(dimPosition) - 1);
            }

            if (otherMultipliers.empty()) {
              // No more-minor dims, just use constant offset.
              int64_t remainder = constSum % modulus;
              gap = (remainder == 0) ? modulus : (modulus - remainder);
            } else {
              // Use minimizeGap to find the best alignment (minimum gap).
              gap =
                  minimizeGap(modulus, otherMultipliers, otherBounds, constSum);
            }
          }

          int64_t stepsNeeded = (gap + targetMultiplier - 1) /
                                targetMultiplier; // Ceiling division
          return checkBoundsAndReturn(stepsNeeded);
        }

        // Target dim not found in expression - unconstrained.
        return -1;
      }
    }

    // Default: recursively analyze LHS with boundValue as the new constraint.
    return analyzeShardResultExprForContiguity(lhs, dimBounds, dimPos,
                                               numGridDims, modulus);
  }

  default:
    return 1;
  }
}

/// Analyzes the stride (coefficient) of a dimension in an affine expression.
/// Returns -1 on error, 0 if dimension doesn't affect this expression,
/// or the stride value.
inline int64_t analyzeExprForDimStride(mlir::AffineExpr expr, unsigned dimPos) {
  // Constant: stride = 0 (doesn't depend on dim).
  if (llvm::isa<mlir::AffineConstantExpr>(expr)) {
    return 0;
  }

  // Dim expression.
  if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    return dimExpr.getPosition() == dimPos ? 1 : 0;
  }

  auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!binOp) {
    return -1; // Cannot analyze.
  }

  auto lhs = binOp.getLHS();
  auto rhs = binOp.getRHS();

  switch (binOp.getKind()) {
  case mlir::AffineExprKind::Add: {
    int64_t lhsStride = analyzeExprForDimStride(lhs, dimPos);
    int64_t rhsStride = analyzeExprForDimStride(rhs, dimPos);
    if (lhsStride < 0 || rhsStride < 0) {
      return -1;
    }
    return lhsStride + rhsStride;
  }

  case mlir::AffineExprKind::Mul: {
    auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    // To simplify analysis, assert that rhs is always a constant here.
    TT_assertv(
        rhsConst,
        "analyzeExprForDimStride: analysis expects MulOp rhs to be a constant");
    int64_t lhsStride = analyzeExprForDimStride(lhs, dimPos);
    if (lhsStride < 0) {
      return -1;
    }
    return lhsStride * rhsConst.getValue();
  }

  case mlir::AffineExprKind::Mod: {
    // Mod doesn't change stride, it just limits the range.
    return analyzeExprForDimStride(lhs, dimPos);
  }

  case mlir::AffineExprKind::FloorDiv:
  case mlir::AffineExprKind::CeilDiv: {
    auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    if (!rhsConst || rhsConst.getValue() <= 0) {
      return -1;
    }

    int64_t lhsStride = analyzeExprForDimStride(lhs, dimPos);
    if (lhsStride < 0) {
      return -1;
    }

    int64_t divisor = rhsConst.getValue();
    // If lhsStride >= divisor, output changes by lhsStride/divisor per step.
    // If lhsStride < divisor, output changes less than once per step (stride
    // 0).
    if (lhsStride < divisor) {
      return 0;
    }
    return lhsStride / divisor;
  }

  default:
    return -1;
  }
}

/// Analyzes stride alignment for shard dimensions. Returns the coalescing
/// factor limited by stride misalignment. Requires exactly one shard result.
inline int64_t analyzeShardDimStrides(mlir::AffineMap map,
                                      mlir::ArrayRef<int64_t> shape,
                                      unsigned numGridDims,
                                      int64_t elemSizeBytes) {
  int numShardDims = shape.size() - numGridDims;
  TT_assert(numShardDims > 0);

  mlir::AffineExpr shardResult = map.getResult(map.getNumResults() - 1);
  size_t innermostDimPos = map.getNumDims() - 1;
  int64_t innermostStride =
      analyzeExprForDimStride(shardResult, innermostDimPos);

  if (innermostStride < 0) {
    return elemSizeBytes;
  }
  // Expect stride for innermost dim to match element size, otherwise it must
  // be strided non-contiguous access. This is only true if the innermost dim
  // is non-unit.
  if (innermostStride > 0 && innermostStride != elemSizeBytes &&
      shape[innermostDimPos] != 1) {
    return elemSizeBytes;
  }

  // If innermost dim has non-unit extent but stride 0 in the shard result,
  // the dimension must affect a grid result instead. Every step of that
  // dimension causes a grid discontinuity, so there's no contiguity.
  if (innermostStride == 0 && shape[innermostDimPos] != 1) {
    return elemSizeBytes;
  }

  // When innermost dim has stride 0 with unit extent (can be skipped) or
  // proper stride, compute the expected accumulated stride.
  int64_t accumulatedStride;
  if (shape[innermostDimPos] == 1) {
    accumulatedStride = elemSizeBytes;
  } else {
    accumulatedStride = innermostStride * shape[innermostDimPos];
  }

  for (int i = map.getNumDims() - 2; i >= static_cast<int>(numGridDims); --i) {
    unsigned dimPos = i;
    if (shape[dimPos] == 1) {
      continue;
    }
    int64_t stride = analyzeExprForDimStride(shardResult, dimPos);
    if (stride < 0) {
      return elemSizeBytes;
    }
    // If a non-unit shard dim has stride 0 in the shard result, it must
    // affect a grid result instead. This causes grid discontinuities when
    // this dim changes, but doesn't prevent coalescing within the innermost
    // contiguous dimensions. Break here rather than returning elemSizeBytes.
    if (stride == 0) {
      break;
    }
    if (stride != accumulatedStride) {
      break;
    }
    accumulatedStride = stride * shape[dimPos];
  }

  return accumulatedStride;
}

/// Finds the minimum change in a dimension needed to change the output value.
/// Used for grid dimension analysis where any change in grid indexing is a
/// discontinuity. Returns -1 if unconstrained, 1 as fallback, or the minimum
/// step count.
inline int64_t analyzeGridResultExprForDiscontinuity(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &dimBounds,
    unsigned dimPos, unsigned numGridDims = 0) {
  // Handle constant expressions - unconstrained (dim doesn't affect output).
  if (llvm::isa<mlir::AffineConstantExpr>(expr)) {
    return -1;
  }

  // Handle dim expressions.
  if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr)) {
    if (dimExpr.getPosition() == dimPos) {
      return 1; // Direct dim reference, any change causes output change.
    }
    return -1; // Different dim, unconstrained.
  }

  auto binOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  if (!binOp) {
    return 1; // Cannot analyze, conservative fallback.
  }

  mlir::AffineExpr lhs = binOp.getLHS();
  mlir::AffineExpr rhs = binOp.getRHS();

  switch (binOp.getKind()) {
  case mlir::AffineExprKind::FloorDiv:
  case mlir::AffineExprKind::CeilDiv: {
    auto rhsConst = llvm::dyn_cast<mlir::AffineConstantExpr>(rhs);
    if (!rhsConst || rhsConst.getValue() <= 0) {
      return 1; // Cannot analyze, conservative fallback.
    }
    int64_t divisor = rhsConst.getValue();

    // Extract multiplier and offset from LHS: looking for M*d + C pattern
    int64_t multiplier = 0;
    bool foundDim = false;

    // Helper to check if steps exceed dim bounds.
    auto checkBoundsAndReturn = [&](int64_t stepsNeeded) -> int64_t {
      int64_t targetDimBound = dimBounds.lookup(dimPos);
      if (stepsNeeded >= targetDimBound) {
        return -1; // Unconstrained - always contiguous within bounds.
      }
      return stepsNeeded;
    };

    // Pattern 1: d_dimPos floorDiv N -> need N steps to change output.
    if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(lhs)) {
      if (dimExpr.getPosition() == dimPos) {
        return checkBoundsAndReturn(divisor);
      }
      // Unrelated dim by itself poses no constraints.
      return -1;
    }

    if (auto lhsMod = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(lhs);
        lhsMod && lhsMod.getKind() == mlir::AffineExprKind::Mod) {

      // Strip away mod, it doesn't impact analysis.
      mlir::AffineExpr fusedExpr;
      if (binOp.getKind() == mlir::AffineExprKind::FloorDiv) {
        fusedExpr = lhsMod.getLHS().floorDiv(divisor);
      } else {
        fusedExpr = lhsMod.getLHS().ceilDiv(divisor);
      }

      return analyzeGridResultExprForDiscontinuity(fusedExpr, dimBounds, dimPos,
                                                   numGridDims);
    }

    // Deal with nested floordiv expressions.
    if (auto lhsBinOp = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(lhs);
        lhsBinOp && lhsBinOp.getKind() == mlir::AffineExprKind::FloorDiv) {

      auto innerRhsConst =
          llvm::dyn_cast<mlir::AffineConstantExpr>(lhsBinOp.getRHS());
      TT_assertv(innerRhsConst, "Nested binary op RHS must be constant");

      int64_t combinedDivisor = innerRhsConst.getValue() * divisor;
      return analyzeGridResultExprForDiscontinuity(
          lhsBinOp.getLHS().floorDiv(combinedDivisor), dimBounds, dimPos,
          numGridDims);
    }

    // Pattern 2: (M * d_dimPos) floorDiv N -> need ceil(N/M) steps.
    if (auto maybeMul = isBinaryMul(lhs)) {
      auto mulRhsConst =
          llvm::dyn_cast<mlir::AffineConstantExpr>(maybeMul.value().getRHS());
      if (!mulRhsConst) {
        return 1; // Mul RHS must be constant, cannot analyze otherwise.
      }
      int64_t mulValue = mulRhsConst.getValue();
      // Now we know RHS is a constant, just check LHS for dim.
      if (exprIsSpecificDimExpr(maybeMul.value().getLHS(), dimPos)) {
        // ceil(divisor / mulValue)
        int64_t stepsNeeded = (divisor + mulValue - 1) / mulValue;
        return checkBoundsAndReturn(stepsNeeded);
      }
      return -1;
    }

    // Pattern 3: (A*d0 + B*d1 + ...) floorDiv N.
    if (!foundDim) {
      if (auto maybeAdd = isBinaryAdd(lhs)) {
        auto sumOperands = collectSumOperands(maybeAdd);

        // If the expression does not contain the target dim, it's
        // unconstrained.
        if (!exprContainsDim(lhs, dimPos)) {
          return -1;
        }

        mlir::AffineExpr targetDimExpr;
        // Track (dimPosition, multiplier) pairs for other dimensions.
        llvm::SmallVector<std::pair<unsigned, int64_t>> otherDimInfo;
        int64_t constSum = 0;

        // Case: ((targetDim floorDiv N) + unrelated dim crap) floorDiv M) ...
        bool foundNestedFloordiv = false;
        bool foundNonDivExprWithTargetDim = false;
        int64_t fusedDivResult = 0;
        for (auto sumOperand : sumOperands) {
          if (auto lhsFloordiv =
                  mlir::dyn_cast<mlir::AffineBinaryOpExpr>(sumOperand);
              lhsFloordiv &&
              (lhsFloordiv.getKind() == mlir::AffineExprKind::FloorDiv)) {

            if (!exprContainsDim(lhsFloordiv.getLHS(), dimPos)) {
              continue;
            }
            foundNestedFloordiv = true;

            mlir::AffineExpr innerRHS = lhsFloordiv.getRHS();
            if (auto innerDivisor =
                    llvm::dyn_cast<mlir::AffineConstantExpr>(innerRHS)) {
              mlir::AffineExpr combinedDivisor = mlir::getAffineConstantExpr(
                  divisor * innerDivisor.getValue(), expr.getContext());

              mlir::AffineExpr fusedExpr =
                  lhsFloordiv.getLHS().floorDiv(combinedDivisor);
              fusedDivResult = analyzeGridResultExprForDiscontinuity(
                  fusedExpr, dimBounds, dimPos, numGridDims);
            }
          } else if (exprContainsDim(sumOperand, dimPos)) {
            foundNonDivExprWithTargetDim = true;
          }
        }
        if (foundNestedFloordiv && !foundNonDivExprWithTargetDim) {
          return fusedDivResult;
        }

        // Identify if dim exists in multiple top-level sum operand exprs.
        int targetDimReferences = 0;
        for (auto sumOperand : sumOperands) {
          bool foundTargetDim = false;
          if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(sumOperand)) {
            if (dimExpr.getPosition() == dimPos) {
              foundTargetDim = true;
            }
          }
          if (foundTargetDim) {
            targetDimReferences++;
          }
        }
        // Cannot analyze expression if the target dim exists in multiple
        // top-level sum operand exprs.
        if (targetDimReferences > 1) {
          return 1;
        }

        for (auto sumOperand : sumOperands) {
          // Case 0: constants - accumulate into constSum.
          if (auto constExpr =
                  llvm::dyn_cast<mlir::AffineConstantExpr>(sumOperand)) {
            constSum += constExpr.getValue();
            continue;
          }

          // Case 1: Plain dim expression (multiplier = 1).
          if (auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(sumOperand)) {
            if (dimExpr.getPosition() == dimPos) {
              if (foundDim) {
                return 1; // Multiple occurrences of target dim, fallback.
              }
              foundDim = true;
              targetDimExpr = sumOperand;
              multiplier = 1;
            } else {
              otherDimInfo.push_back({dimExpr.getPosition(), 1});
            }
            continue;
          }

          // Case 2: Mul expression (dim * const).
          if (auto mulOp = isBinaryMul(sumOperand)) {
            auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(mulOp->getLHS());
            auto constExpr =
                llvm::dyn_cast<mlir::AffineConstantExpr>(mulOp->getRHS());

            if (!dimExpr || !constExpr) {
              // lhs is binary op expr, can only analyze if this contains the
              // only reference to the target dim.
              if (exprContainsDim(sumOperand, dimPos)) {
                // Worst case assume that any change in the lhs expr will change
                // the output.
                return analyzeGridResultExprForDiscontinuity(
                    sumOperand, dimBounds, dimPos, numGridDims);
              }
              // Don't try to analyze this for now.
              return 1;
            }

            int64_t mulValue = constExpr.getValue();

            if (dimExpr.getPosition() == dimPos) {
              foundDim = true;
              targetDimExpr = sumOperand;
              multiplier = mulValue;
            } else {
              otherDimInfo.push_back({dimExpr.getPosition(), mulValue});
            }
            continue;
          }

          // Anything else: fallback.
          return 1;
        }

        // If we found the target dim, compute worst-case discontinuity.
        if (foundDim && multiplier > 0) {
          if (constSum < 0) {
            return 1; // Negative constants not analyzed, fallback.
          }

          int64_t gap;
          if (otherDimInfo.empty()) {
            // Simple case: just the target dim and a constant offset.
            // Gap is distance from constSum to the next divisor boundary.
            int64_t remainder = constSum % divisor;
            gap = (remainder == 0) ? divisor : (divisor - remainder);
          } else {
            // Build multipliers and bounds arrays from tracked dimension info.
            // Include ALL other dimensions to find worst-case gap:
            //   - Grid dims: fixed per transfer, need worst case across grid
            //   - Outer shard dims: fixed during inner iteration, need worst
            //   case
            //   - Inner shard dims: vary within iteration
            // All must be considered to find the minimum gap that produces
            // the worst-case contiguity bound.
            llvm::SmallVector<int64_t> otherMultipliers;
            llvm::SmallVector<int64_t> otherBounds;
            for (auto [dimPosition, mul] : otherDimInfo) {
              otherMultipliers.push_back(mul);
              // Bound is (dimSize - 1) since dim ranges from 0 to dimSize-1
              otherBounds.push_back(dimBounds.lookup(dimPosition) - 1);
            }

            if (otherMultipliers.empty()) {
              // No more-minor dims, just use constant offset.
              int64_t remainder = constSum % divisor;
              gap = (remainder == 0) ? divisor : (divisor - remainder);
            } else {
              // Use minimizeGap to find the best alignment (minimum gap).
              gap =
                  minimizeGap(divisor, otherMultipliers, otherBounds, constSum);
            }
          }

          int64_t stepsNeeded =
              (gap + multiplier - 1) / multiplier; // Ceiling division
          auto it = checkBoundsAndReturn(stepsNeeded);
          return it;
        }
      }
    }

    // Cannot analyze, conservative fallback.
    return 1;
  }

  case mlir::AffineExprKind::Add: {
    int64_t lhsResult = analyzeGridResultExprForDiscontinuity(
        lhs, dimBounds, dimPos, numGridDims);
    int64_t rhsResult = analyzeGridResultExprForDiscontinuity(
        rhs, dimBounds, dimPos, numGridDims);

    // Combine results: take the minimum of constrained values.
    // -1 means unconstrained, so we filter those out.
    if (lhsResult == -1 && rhsResult == -1) {
      return -1; // Both unconstrained.
    }
    if (lhsResult == -1) {
      return rhsResult;
    }
    if (rhsResult == -1) {
      return lhsResult;
    }
    return std::gcd(lhsResult, rhsResult);
  }

  case mlir::AffineExprKind::Mod:
  case mlir::AffineExprKind::Mul: {
    // If a mul expression contains the target dim, it implies any change in the
    // dim will change the output unless this op is the LHS of a floorDiv or
    // ceilDiv expression that contains the target dim.
    // If not, by definition it's unconstrained.
    if (!exprContainsDim(lhs, dimPos)) {
      return -1;
    }
    return analyzeGridResultExprForDiscontinuity(lhs, dimBounds, dimPos,
                                                 numGridDims);
  }

  default:
    return 1; // Cannot analyze, conservative fallback.
  }
}

/// Analyzes contiguity for a single shard dimension (position >= numGridDims).
/// Returns 1 on error or the maximum coalescable run length for this dim.
/// Requires exactly one shard result (the last result in the map).
inline int64_t analyzeSingleShardDimContiguity(mlir::AffineMap map,
                                               mlir::ArrayRef<int64_t> shape,
                                               unsigned numGridDims,
                                               unsigned shardDimIdx) {
  unsigned dimPos = numGridDims + shardDimIdx;
  unsigned numGridResults = map.getNumResults() - 1;

  // The dim bounds are derived from the shape at each position.
  llvm::DenseMap<int, int64_t> dimBounds;
  for (unsigned i = 0; i < map.getNumDims(); ++i) {
    dimBounds[static_cast<int>(i)] = shape[i];
  }

  int64_t dimContiguity = -1; // Start unconstrained.
  auto results = map.getResults();

  for (size_t idx = 0; idx < results.size(); ++idx) {
    mlir::AffineExpr resultExpr = results[idx];
    int64_t exprBound;
    if (idx >= numGridResults) {
      exprBound = analyzeShardResultExprForContiguity(resultExpr, dimBounds,
                                                      dimPos, numGridDims);
    } else {
      exprBound = analyzeGridResultExprForDiscontinuity(resultExpr, dimBounds,
                                                        dimPos, numGridDims);
    }

    // Combine bounds using GCD.
    if (exprBound != -1) {
      if (dimContiguity == -1) {
        dimContiguity = exprBound;
      } else {
        dimContiguity = std::gcd(dimContiguity, exprBound);
      }
    }
  }

  // Replace -1 (unconstrained) with the actual shape extent for this dim.
  if (dimContiguity == -1) {
    dimContiguity = shape[dimPos];
  }

  return dimContiguity;
}

/// Computes a combined coalescing factor by analyzing all shard dimensions
/// from most minor to least minor, checking both stride alignment and
/// contiguity constraints. Returns the maximum coalescable run length.
/// Requires exactly one shard result (the last result in the map).
inline int64_t analyzeShardDimContiguity(mlir::AffineMap map,
                                         mlir::ArrayRef<int64_t> shape,
                                         unsigned numGridDims,
                                         int64_t elemSizeBytes) {
  TT_assert(elemSizeBytes > 0);
  TT_assertv(shape.size() == map.getNumDims(),
             "Shape size must match number of map dimensions");
  TT_assertv(shape.size() % 2 == 0u, "Shape rank must be even");
  TT_assertv(map.getNumResults() == numGridDims + 1,
             "Map must have exactly one shard result");

  unsigned numShardDims = shape.size() - numGridDims;

  if (numShardDims == 0) {
    return 1; // No shard dims means trivially contiguous.
  }

  // Do initial pass of simplification; all remaining mod/floordiv expressions
  // actually do something.
  mlir::AffineMap simplifiedMap =
      simplifyAffineMapWithRangeAnalysis(simplifyZeroFloorDiv(map), shape);

  // Analyze each shard dim and store results.
  llvm::SmallVector<int64_t> dimContiguities;
  dimContiguities.reserve(numShardDims);

  for (unsigned shardDimIdx = 0; shardDimIdx < numShardDims; ++shardDimIdx) {
    int64_t contiguity = analyzeSingleShardDimContiguity(
        simplifiedMap, shape, numGridDims, shardDimIdx);
    dimContiguities.push_back(contiguity);
  }

  // Iterate from most minor to least minor shard dim.
  // Accumulate the product of fully contiguous minor dims.
  int64_t coalescingFactor = 1;
  for (int dimIdx = numShardDims - 1; dimIdx >= 0; --dimIdx) {
    unsigned dimPos = numGridDims + dimIdx;
    int64_t dimExtent = shape[dimPos];
    int64_t dimContiguity = dimContiguities[dimIdx];

    coalescingFactor *= dimContiguity;
    if (dimContiguity < dimExtent) {
      // Stop accumulating the coalescing factor once non-contiguous dim is
      // encountered.
      break;
    }
  }
  // Check stride alignment. This fundamentally limits the coalescing factor.
  auto strideLimit =
      analyzeShardDimStrides(simplifiedMap, shape, numGridDims, elemSizeBytes);
  TT_assertv(strideLimit % elemSizeBytes == 0,
             "strideLimit must be divisible by elemSizeBytes");
  strideLimit /= elemSizeBytes;

  if (strideLimit < coalescingFactor) {
    return std::gcd(coalescingFactor, strideLimit);
  }

  return coalescingFactor;
}

} // namespace ttmlir::utils

#endif // TTMLIR_AFFINEMAPANALYSIS_H
