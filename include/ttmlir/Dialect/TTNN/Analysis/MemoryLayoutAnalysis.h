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
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"

#include <vector>

namespace mlir::tt::ttnn {

struct MemoryLayoutAnalysisInput {
  const TensorTypeLayoutsMap *tensorTypePossibleLayouts;
  llvm::DenseMap<Operation *, std::vector<OpConfig>> legalConfigs;
  llvm::DenseSet<Edge> overrideReshardEdges;
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout;

  MemoryLayoutAnalysisPolicyType policy;

  MemoryLayoutAnalysisInput() : legalConfigs() {}

  MemoryLayoutAnalysisInput(
      const TensorTypeLayoutsMap *tensorTypePossibleLayouts,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      const llvm::DenseSet<Edge> &overrideReshardEdges,
      const llvm::StringMap<OutputLayoutOverrideParams> &overrideOutputLayout,
      MemoryLayoutAnalysisPolicyType policy)
      : tensorTypePossibleLayouts(tensorTypePossibleLayouts),
        legalConfigs(legalConfigs), overrideReshardEdges(overrideReshardEdges),
        overrideOutputLayout(overrideOutputLayout), policy(policy) {}

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
  std::vector<Operation *> spillToL1InterleavedOps;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule;

  MemoryLayoutAnalysisResult()
      : legalConfigs(), memReconfigEntryMap(), spillToDramOps(),
        spillToL1InterleavedOps(), schedule() {}

  MemoryLayoutAnalysisResult(
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      const llvm::DenseMap<Edge, MemReconfigEntry> &memReconfigEntryMap,
      const std::vector<Operation *> &spillToDramOps,
      const std::vector<Operation *> &spillToL1InterleavedOps)
      : legalConfigs(legalConfigs), memReconfigEntryMap(memReconfigEntryMap),
        spillToDramOps(spillToDramOps),
        spillToL1InterleavedOps(spillToL1InterleavedOps) {}
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
