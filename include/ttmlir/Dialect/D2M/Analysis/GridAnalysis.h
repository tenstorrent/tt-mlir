// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_ANALYSIS_GRIDANALYSIS_H
#define TTMLIR_DIALECT_D2M_ANALYSIS_GRIDANALYSIS_H

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace mlir::tt::d2m {

/// Configuration for GridAnalysis grid computation.
struct GridConfig {
  llvm::SmallVector<int64_t> targetGridShape;
  bool ttnnMode;
};

/// Classification of how a GenericOp operand should be updated during the
/// grid selection transform phase.
enum class OperandUpdateKind {
  ToLayout,
  TTNNTensor,
  Empty,
  ViewLayout,
  CompositeView,
  None,
};

/// Per-operand analysis result describing the chosen grid and which op needs
/// updating.
struct OperandGridInfo {
  unsigned operandIndex;
  llvm::SmallVector<int64_t> selectedGrid;
  llvm::SmallVector<int64_t> targetGrid; // Per-operand target grid used for
                                         // grid computation and alignment.
  OperandUpdateKind updateKind;

  // Op handles — only the relevant one is set based on updateKind.
  d2m::ToLayoutOp toLayoutOp;
  Value ttnnOperand;
  d2m::EmptyOp emptyOp;
  d2m::ViewLayoutOp viewLayoutOp;
  d2m::CompositeViewOp compositeViewOp;

  // For ViewLayout operands whose input is a ToLayoutOp: the ToLayoutOp's
  // independently computed optimal grid and op handle.
  llvm::SmallVector<int64_t> toLayoutGrid;
  d2m::ToLayoutOp behindViewToLayoutOp;
};

/// Per-GenericOp analysis result containing all grid decisions.
struct GenericGridAnalysisResult {
  llvm::SmallVector<OperandGridInfo, 4> operandInfos;
  llvm::SmallVector<llvm::SmallVector<int64_t>> normalizedOperandGrids;
  bool ttnnMode;

  // The effective target grid used for grid computation. Defaults to the full
  // device grid; per-op/operand decisions can constrain it (e.g. square for
  // matmul k-dim).
  llvm::SmallVector<int64_t> effectiveTargetGrid;
};

/// Module-level analysis that computes optimal grid assignments for all
/// d2m.generic ops before any IR modification.
///
/// Usage pattern (matches BlockFactorAnalysis):
///   GridAnalysis gridAnalysis(moduleOp, options);
///   for (auto genericOp : ...) {
///     const auto *result = gridAnalysis.lookup(genericOp);
///     // apply transforms using result
///   }
struct GridAnalysis {
  struct Options {
    llvm::SmallVector<int64_t> deviceGridShape;
    bool ttnnMode = false;
  };

  GridAnalysis(Operation *moduleOp, const Options &opts);

  /// Look up the pre-computed grid analysis for a generic op.
  /// Returns nullptr if the generic was skipped.
  const GenericGridAnalysisResult *lookup(GenericOp genericOp) const;

private:
  /// Analyze a single GenericOp and compute grid decisions for all operands.
  GenericGridAnalysisResult analyzeGenericOp(GenericOp genericOp,
                                             const GridConfig &config);

  /// Compute the target grid shape for a generic, accounting for spatial
  /// region grid ranges.
  llvm::SmallVector<int64_t> getTargetGridShape(GenericOp genericOp) const;

  /// Normalize operand grids within a generic to ensure consistency across
  /// operands sharing loop dimensions. Physical shapes are required to ensure
  /// promoted grid factors evenly divide all affected operands.
  static llvm::SmallVector<llvm::SmallVector<int64_t>>
  normalizeOperandGridsForGeneric(
      GenericOp genericOp,
      ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
      ArrayRef<llvm::SmallVector<int64_t>> physicalShapes);

  /// Check if an operand traces back to a TTNNMetalLayoutCastOp through
  /// ViewLayoutOp chains.
  static bool isTTNNOperand(Value operand);

  llvm::DenseMap<Operation *, std::unique_ptr<GenericGridAnalysisResult>>
      results;
  llvm::SmallVector<int64_t> deviceGridShape;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_GRIDANALYSIS_H
