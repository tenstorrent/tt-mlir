// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MEMORYLAYOUTANALYSIS_H

#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/Utils/MemoryLayoutAnalysisParams.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"

#include <vector>

namespace mlir::tt::ttnn {

struct MemoryLayoutAnalysisInput {
  const TensorTypeLayoutsMap *tensorTypePossibleLayouts;
  llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;
  unsigned usableL1CacheSize = 0;
  llvm::DenseSet<Edge> overrideReshardEdges;
  llvm::DenseSet<Operation *> rowMajorOutputOps;

  MemoryLayoutAnalysisPolicyType policy;

  MemoryLayoutAnalysisInput() : legalConfigs() {}

  MemoryLayoutAnalysisInput(
      const TensorTypeLayoutsMap *tensorTypePossibleLayouts,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      unsigned usableL1CacheSize,
      const llvm::DenseSet<Edge> &overrideReshardEdges,
      const llvm::DenseSet<Operation *> &rowMajorOutputOps,
      MemoryLayoutAnalysisPolicyType policy)
      : tensorTypePossibleLayouts(tensorTypePossibleLayouts),
        legalConfigs(legalConfigs), usableL1CacheSize(usableL1CacheSize),
        overrideReshardEdges(overrideReshardEdges),
        rowMajorOutputOps(rowMajorOutputOps), policy(policy) {}

  bool operator==(const MemoryLayoutAnalysisInput &rhs) const {
    return legalConfigs == rhs.legalConfigs;
  }

  bool operator!=(const MemoryLayoutAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct MemoryLayoutAnalysisResult {
  llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;
  llvm::DenseMap<Edge, MemReconfigEntry> memReconfigEntryMap;
  std::vector<Operation *> spillToDramOps;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule;

  MemoryLayoutAnalysisResult()
      : legalConfigs(), memReconfigEntryMap(), spillToDramOps(), schedule() {}

  MemoryLayoutAnalysisResult(
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      const llvm::DenseMap<Edge, MemReconfigEntry> &memReconfigEntryMap,
      const std::vector<Operation *> &spillToDramOps)
      : legalConfigs(legalConfigs), memReconfigEntryMap(memReconfigEntryMap),
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
