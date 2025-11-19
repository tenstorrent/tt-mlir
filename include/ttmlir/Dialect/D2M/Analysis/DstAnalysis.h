// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSIS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::tt::d2m {

/// Result of DST analysis for a single operation.
///
/// Contains the computed DST slice requirements and diagnostic information
/// about whether the operation fits within available capacity.
struct DstAnalysisResult {
  /// Number of DST slices required by this operation.
  unsigned numSlicesRequired = 0;

  /// Whether the requirements are valid (fit within available capacity).
  /// If false, check failureReason for diagnostic information.
  bool isValid = true;

  /// Optional breakdown of slice requirements per sub-operation.
  /// Maps each operation to its assigned color/slice number.
  /// Useful for detailed diagnostics and debugging.
  llvm::DenseMap<Operation *, unsigned> operationSlices;

  /// Identified DST accesses that need allocation.
  /// Each pair contains the operation and its index within the access sequence.
  /// This allows transformation passes to reuse the analysis results without
  /// re-identifying accesses.
  llvm::SmallVector<std::pair<Operation *, int64_t>> dstAccesses;

  /// Human-readable explanation if analysis failed or requirements exceed
  /// capacity.
  std::optional<std::string> failureReason;
};

/// Abstract interface for DST analysis strategies.
///
/// This interface enables multiple analysis implementations (naive counting,
/// graph coloring, etc.) to be plugged into the same analysis framework.
/// Each strategy computes DST slice requirements differently but returns
/// results in a uniform format.
class DstAnalysis {
public:
  virtual ~DstAnalysis() = default;

  /// Analyze DST requirements for the given operation.
  ///
  /// \param op The operation to analyze (typically a d2m::GenericOp).
  /// \return DstAnalysisResult containing slice count and validity information.
  virtual DstAnalysisResult analyze(Operation *op) = 0;

  /// Get human-readable name of this analysis strategy.
  ///
  /// Used for diagnostics, debugging, and command-line option descriptions.
  /// \return Strategy name (e.g., "basic", "graph-coloring", "greedy").
  virtual llvm::StringRef getStrategyName() const = 0;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_DSTANALYSIS_H
