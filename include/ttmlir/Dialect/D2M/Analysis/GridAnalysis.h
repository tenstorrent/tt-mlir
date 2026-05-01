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

/// Complete grid decision for one tensor value.
///
/// selectedGrid is the tensor grid stored in the layout. physicalGrid is the
/// 2D worker-grid extent used to place a virtual grid. layoutGrid is the grid
/// used when computing grid-aware dim alignments for this selected grid.
struct GridDecision {
  llvm::SmallVector<int64_t> selectedGrid;
  llvm::SmallVector<int64_t> targetGrid;
  llvm::SmallVector<int64_t> physicalGrid;
  llvm::SmallVector<int64_t> layoutGrid;

  bool empty() const { return selectedGrid.empty(); }
  bool isVirtual() const { return selectedGrid != physicalGrid; }
};

/// Per-operand analysis result describing the chosen grid for a GenericOp
/// operand. The concrete update strategy is recovered at apply time from the
/// operand's defining op.
struct OperandGridInfo {
  Value operand;
  GridDecision grid;

  // Set only when the operand is a ViewLayoutOp whose input is a ToLayoutOp
  // that should have its grid optimized independently. Carries the optimal
  // grid for that upstream ToLayoutOp's own tensor shape. Empty otherwise.
  GridDecision viewSourceGrid;
};

/// Effective target grid range for a GenericOp.
/// Shape is the 2D extent, offset is the 2D start coordinate on device grid.
struct EffectiveTargetGridRange {
  // The shape is always expected to be 2D.
  llvm::SmallVector<int64_t> shape;
  // Offset defaults to origin when the target range is the full device grid.
  llvm::SmallVector<int64_t> offset = {0, 0};
};

/// Per-GenericOp analysis result containing all grid decisions.
struct GenericGridAnalysisResult {
  llvm::SmallVector<OperandGridInfo, 4> operandInfos;
  llvm::SmallVector<llvm::SmallVector<int64_t>> normalizedOperandGrids;
  // The effective target grid range for this generic: full device grid by
  // default, or the range scoped by an enclosing d2m.spatial region.
  EffectiveTargetGridRange effectiveTargetGridRange;
};

/// Module-level analysis that computes optimal grid assignments for all
/// d2m.generic ops before any IR modification.
///
/// Usage pattern (matches BlockFactorAnalysis):
///   GridAnalysis gridAnalysis(moduleOp, deviceGridShape, ttnnMode);
///   for (auto genericOp : ...) {
///     const auto *result = gridAnalysis.lookup(genericOp);
///     // apply transforms using result
///   }
struct GridAnalysis {
  GridAnalysis(Operation *moduleOp, ArrayRef<int64_t> deviceGridShape,
               bool ttnnMode);

  /// Look up the pre-computed grid analysis for a generic op.
  /// Returns nullptr if the generic was skipped.
  const GenericGridAnalysisResult *lookup(GenericOp genericOp) const;

  /// Check if an operand traces back to a TTNNMetalLayoutCastOp through
  /// ViewLayoutOp chains.
  static bool isTTNNOperand(Value operand);

private:
  /// Analyze a single GenericOp and compute grid decisions for all operands.
  GenericGridAnalysisResult
  analyzeGenericOp(GenericOp genericOp,
                   const EffectiveTargetGridRange &effectiveTargetGridRange);

  /// Compute the effective target grid range for a generic, accounting for
  /// spatial region grid ranges.
  EffectiveTargetGridRange getTargetGridRange(GenericOp genericOp) const;

  /// Normalize operand grids within a generic to ensure consistency across
  /// operands sharing loop dimensions. Physical shapes are required to ensure
  /// promoted grid factors evenly divide all affected operands.
  static llvm::SmallVector<llvm::SmallVector<int64_t>>
  normalizeOperandGridsForGeneric(
      GenericOp genericOp,
      ArrayRef<llvm::SmallVector<int64_t>> optimalOperandGrids,
      ArrayRef<llvm::SmallVector<int64_t>> physicalShapes);

  llvm::DenseMap<Operation *, std::unique_ptr<GenericGridAnalysisResult>>
      results;
  llvm::SmallVector<int64_t> deviceGridShape;
  bool ttnnMode;
};

} // namespace mlir::tt::d2m

#endif // TTMLIR_DIALECT_D2M_ANALYSIS_GRIDANALYSIS_H
