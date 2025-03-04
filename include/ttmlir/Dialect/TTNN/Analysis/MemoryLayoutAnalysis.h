// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Utils/MemoryLayoutAnalysisParams.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <vector>

namespace mlir::tt::ttnn {

struct MemoryLayoutAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>> legalLayouts;
  unsigned usableL1CacheSize = 0;
  std::unordered_set<Edge> overrideReshardEdges;
  MemoryLayoutAnalysisPolicyType policy;

  MemoryLayoutAnalysisInput() : legalLayouts() {}

  MemoryLayoutAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>>
          &legalLayouts,
      unsigned usableL1CacheSize,
      const std::unordered_set<Edge> &overrideReshardEdges,
      MemoryLayoutAnalysisPolicyType policy)
      : legalLayouts(legalLayouts), usableL1CacheSize(usableL1CacheSize),
        overrideReshardEdges(overrideReshardEdges), policy(policy) {}

  bool operator==(const MemoryLayoutAnalysisInput &rhs) const {
    return legalLayouts == rhs.legalLayouts;
  }

  bool operator!=(const MemoryLayoutAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct MemoryLayoutAnalysisResult {
  llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>> legalLayouts;
  std::unordered_set<Edge> memReconfigEdges;
  std::vector<Operation *> spillToDramOps;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule;

  MemoryLayoutAnalysisResult()
      : legalLayouts(), memReconfigEdges(), spillToDramOps(), schedule() {}

  MemoryLayoutAnalysisResult(
      const llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>>
          &legalLayouts,
      const std::unordered_set<Edge> &memReconfigEdges,
      const std::vector<Operation *> &spillToDramOps)
      : legalLayouts(legalLayouts), memReconfigEdges(memReconfigEdges),
        spillToDramOps(spillToDramOps) {}
};

// Analyze and determine which parts of the model graph can be pushed to L1
// memory. Produce L1 chain configs consisting of OPs leaving results in L1.
//
class MemoryLayoutAnalysis : public TTNNAnalysis<MemoryLayoutAnalysisInput,
                                                 MemoryLayoutAnalysisResult> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override;
  std::vector<L1ChainConfig> l1ChainConfigs;

public:
  MemoryLayoutAnalysis(Operation *op) : TTNNAnalysis(op) {}
};
} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSIS_H
