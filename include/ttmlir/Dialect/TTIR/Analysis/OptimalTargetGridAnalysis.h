// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_OPTIMALTARGETGRIDANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_OPTIMALTARGETGRIDANALYSIS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Analysis/TTIRAnalysis.h"

namespace mlir::tt::ttir {

struct OptimalTargetGridAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<GridAttr>> legalGrids;

  OptimalTargetGridAnalysisInput() : legalGrids() {}

  OptimalTargetGridAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<GridAttr>> &&legalGrids)
      : legalGrids(std::move(legalGrids)) {}

  bool operator==(const OptimalTargetGridAnalysisInput &rhs) const {
    return legalGrids == rhs.legalGrids;
  }

  bool operator!=(const OptimalTargetGridAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

// Determine optimal target grid size for each op.
//
class OptimalTargetGridAnalysis
    : public TTIRAnalysis<OptimalTargetGridAnalysisInput,
                          llvm::DenseMap<Operation *, GridAttr>> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

public:
  OptimalTargetGridAnalysis(Operation *op) : TTIRAnalysis(op) {}
};
} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_OPTIMALTARGETGRIDANALYSIS_H
