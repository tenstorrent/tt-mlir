// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_OPCONFIGANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_OPCONFIGANALYSIS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Analysis/TTIRAnalysis.h"

namespace mlir::tt::ttir {

struct OpConfigAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<LayoutAttr>> legalGrids;

  OpConfigAnalysisInput() : legalGrids() {}

  OpConfigAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &&legalGrids)
      : legalGrids(std::move(legalGrids)) {}

  OpConfigAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalGrids)
      : legalGrids(legalGrids) {}

  bool operator==(const OpConfigAnalysisInput &rhs) const {
    return legalGrids == rhs.legalGrids;
  }

  bool operator!=(const OpConfigAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

// Determine optimal configuration for each op.
//
class OpConfigAnalysis
    : public TTIRAnalysis<OpConfigAnalysisInput,
                          llvm::DenseMap<Operation *, LayoutAttr>> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

public:
  OpConfigAnalysis(Operation *op) : TTIRAnalysis(op) {}
};
} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_OPCONFIGANALYSIS_H
