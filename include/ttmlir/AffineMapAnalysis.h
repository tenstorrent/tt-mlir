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

/// Simplifies an affine map by eliminating redundant mod/floordiv operations
/// using range analysis. If the upper bound of an operand is less than the
/// modulus, the mod is removed; if less than the divisor, floordiv becomes 0.
/// When performEquivalenceCheck is true, samples the domain to verify the
/// simplified map is equivalent; returns the original map if not.
mlir::AffineMap
simplifyAffineMapWithRangeAnalysis(mlir::AffineMap map,
                                   mlir::ArrayRef<int64_t> dimBounds,
                                   bool performEquivalenceCheck);

/// Recursively analyzes contiguity of an affine expression to find the maximum
/// coalescable run. Returns UnconstrainedBound if unconstrained,
/// UnanalyzableBound if unanalyzable (treated as ConstrainedBound{1}), or
/// ConstrainedBound with the contiguity bound.
ContiguityBound analyzeShardResultExprForContiguity(
    mlir::AffineExpr expr, const llvm::DenseMap<int, int64_t> &dimBounds,
    unsigned dimPos, unsigned numGridDims = 0,
    std::optional<int64_t> parentModulus = std::nullopt);

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
