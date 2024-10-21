// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGANALYSIS_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"

namespace mlir::tt::ttnn {

struct OpConfigAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> legalGrids;

  OpConfigAnalysisInput() : legalGrids() {}

  OpConfigAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
          &&legalGrids)
      : legalGrids(std::move(legalGrids)) {}

  OpConfigAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
          &legalGrids)
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
    : public TTNNAnalysis<OpConfigAnalysisInput,
                          llvm::DenseMap<Operation *, tt::LayoutAttr>> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

public:
  OpConfigAnalysis(Operation *op) : TTNNAnalysis(op) {}
};
} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGANALYSIS_H
