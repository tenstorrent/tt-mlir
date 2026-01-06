// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_AFFINEMAPANALYSIS_H
#define TTMLIR_AFFINEMAPANALYSIS_H

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>
#include <optional>
#include <variant>

namespace ttmlir::utils {

/// Represents an unconstrained contiguity bound (equivalent to -1).
struct UnconstrainedBound {};

/// Represents a constrained contiguity bound with a positive non-zero value.
struct ConstrainedBound {
  int64_t value;
  explicit ConstrainedBound(int64_t v) : value(v) {}
};

/// Represents an unanalyzable expression (treated as ConstrainedBound with
/// value 1).
struct UnanalyzableBound {};

/// Variant type for contiguity analysis results.
using ContiguityBound =
    std::variant<UnconstrainedBound, ConstrainedBound, UnanalyzableBound>;

/// Helper functions to check variant type.
inline bool isUnconstrainedBound(const ContiguityBound &bound) {
  return std::holds_alternative<UnconstrainedBound>(bound);
}

inline bool isUnanalyzableBound(const ContiguityBound &bound) {
  return std::holds_alternative<UnanalyzableBound>(bound);
}

inline bool isConstrainedBound(const ContiguityBound &bound) {
  return std::holds_alternative<ConstrainedBound>(bound);
}

/// Extracts bound value from ContiguityBound variant.
/// Returns std::nullopt for UnconstrainedBound, 1 for UnanalyzableBound
/// (treated as ConstrainedBound{1}), or the value for ConstrainedBound.
inline std::optional<int64_t> getBoundValue(const ContiguityBound &bound) {
  if (isUnconstrainedBound(bound)) {
    return std::nullopt;
  }
  if (isUnanalyzableBound(bound)) {
    return 1; // Treat as ConstrainedBound{1}.
  }
  return std::get<ConstrainedBound>(bound).value;
}

inline std::optional<mlir::AffineDimExpr> isDimExpr(mlir::AffineExpr expr) {
  auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr);
  return dimExpr ? std::optional<mlir::AffineDimExpr>(dimExpr) : std::nullopt;
}

std::optional<int64_t> getSumOfModuli(mlir::AffineExpr expr);

mlir::AffineExpr simplifyZeroFloorDivExpr(mlir::AffineExpr expr);

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

std::optional<int64_t> getExprUpperBound(mlir::AffineExpr expr,
                                         mlir::ArrayRef<int64_t> dimBounds);

mlir::AffineExpr
simplifyAffineExprWithRangeAnalysis(mlir::AffineExpr expr,
                                    mlir::ArrayRef<int64_t> dimBounds);

/// Simplifies an affine map by eliminating redundant mod/floordiv operations
/// using range analysis. If the upper bound of an operand is less than the
/// modulus, the mod is removed; if less than the divisor, floordiv becomes 0.
/// When performEquivalenceCheck is true, samples the domain to verify the
/// simplified map is equivalent; returns the original map if not.
mlir::AffineMap
simplifyAffineMapWithRangeAnalysis(mlir::AffineMap map,
                                   mlir::ArrayRef<int64_t> dimBounds,
                                   bool performEquivalenceCheck);

inline mlir::AffineExpr getIfBinaryAdd(mlir::AffineExpr expr) {
  auto lhsBinExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  return (lhsBinExpr && lhsBinExpr.getKind() == mlir::AffineExprKind::Add)
             ? lhsBinExpr
             : mlir::AffineExpr{};
}

inline mlir::AffineExpr getIfBinaryMul(mlir::AffineExpr expr) {
  auto lhsBinExpr = llvm::dyn_cast<mlir::AffineBinaryOpExpr>(expr);
  return (lhsBinExpr && lhsBinExpr.getKind() == mlir::AffineExprKind::Mul)
             ? lhsBinExpr
             : mlir::AffineExpr{};
}

void collectSumOperandsImpl(mlir::AffineExpr expr,
                            llvm::SmallVectorImpl<mlir::AffineExpr> &results);

mlir::SmallVector<mlir::AffineExpr> collectSumOperands(mlir::AffineExpr expr);

/// Computes the minimal non-zero positive difference between any possible sum
/// of products (given multipliers, bounded variables, and a constant offset)
/// and any integer multiple of `targetSum`. The result is the smallest such
/// difference greater than zero, effectively finding the least nonzero gap
/// between possible sum values and the set of all integer multiples of
/// `targetSum`.
/// This is closely related to a similar algorithm for the change-making problem
/// or subset sum, and is a tweaked version of Bellman's dynamic programming
/// solution.
int64_t minimizeGap(int64_t targetSum, llvm::ArrayRef<int64_t> multipliers,
                    llvm::ArrayRef<int64_t> bounds, int64_t constOffset);

bool exprContainsDim(mlir::AffineExpr expr, unsigned dimPos);

inline bool exprIsSpecificDimExpr(mlir::AffineExpr expr, unsigned dimPos) {
  return llvm::isa<mlir::AffineDimExpr>(expr) &&
         llvm::dyn_cast<mlir::AffineDimExpr>(expr).getPosition() == dimPos;
}

/// Recursively analyzes contiguity of an affine expression to find the maximum
/// coalescable run. Returns UnconstrainedBound if unconstrained,
/// UnanalyzableBound if unanalyzable (treated as ConstrainedBound{1}), or
/// ConstrainedBound with the contiguity bound.
ContiguityBound analyzeShardResultExprForContiguity(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &dimBounds,
    unsigned dimPos, unsigned numGridDims = 0,
    std::optional<int64_t> parentModulus = std::nullopt);

/// Analyzes the stride (coefficient) of a dimension in an affine expression.
/// Returns -1 on error, 0 if dimension doesn't affect this expression,
/// or the stride value.
int64_t analyzeExprForDimStride(mlir::AffineExpr expr, unsigned dimPos);

/// Analyzes stride alignment for shard dimensions. Returns std::nullopt if
/// unanalyzable, or the coalescing factor limited by stride misalignment.
/// Requires exactly one shard result.
std::optional<size_t> analyzeShardDimStrides(mlir::AffineMap map,
                                             mlir::ArrayRef<int64_t> shape,
                                             unsigned numGridDims,
                                             int64_t elemSizeBytes);

/// Finds the minimum change in a dimension needed to change the output value.
/// Used for grid dimension analysis where any change in grid indexing is a
/// discontinuity. Returns UnconstrainedBound if unconstrained,
/// UnanalyzableBound if unanalyzable (treated as ConstrainedBound{1}), or
/// ConstrainedBound with the minimum step count.
ContiguityBound analyzeGridResultExprForDiscontinuity(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &dimBounds,
    unsigned dimPos, unsigned numGridDims = 0);

/// Analyzes contiguity for a single shard dimension (position >= numGridDims).
/// Returns std::nullopt if unanalyzable, or the maximum coalescable run length
/// for this dim. Requires exactly one shard result (the last result in the
/// map).
std::optional<int64_t>
computeCoalescingFactorForShardDim(mlir::AffineMap map,
                                   mlir::ArrayRef<int64_t> shape,
                                   unsigned numGridDims, unsigned shardDimIdx);

/// Computes a combined coalescing factor by analyzing all shard dimensions
/// from most minor to least minor, checking both stride alignment and
/// contiguity constraints. Returns the maximum coalescable run length.
/// Requires exactly one shard result (the last result in the map).
size_t computeCoalescingFactorAnalytically(mlir::AffineMap map,
                                           mlir::ArrayRef<int64_t> shape,
                                           unsigned numGridDims,
                                           size_t elemSizeBytes);

} // namespace ttmlir::utils

#endif // TTMLIR_AFFINEMAPANALYSIS_H
