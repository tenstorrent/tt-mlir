// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

struct OpConfigAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;

  OpConfigAnalysisInput() : legalConfigs() {}

  OpConfigAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &&legalConfigs)
      : legalConfigs(std::move(legalConfigs)) {}

  OpConfigAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs)
      : legalConfigs(legalConfigs) {}

  bool operator==(const OpConfigAnalysisInput &rhs) const {
    return legalConfigs == rhs.legalConfigs;
  }

  bool operator!=(const OpConfigAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

// Determine optimal configuration for each op.
//
class OpConfigAnalysis
    : public TTNNAnalysis<OpConfigAnalysisInput,
                          llvm::DenseMap<Operation *, OpConfig>> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override;

public:
  OpConfigAnalysis(Operation *op) : TTNNAnalysis(op) {}
};
} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_OPCONFIGANALYSIS_H
