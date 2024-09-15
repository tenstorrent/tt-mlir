// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_SHARDINGANALYSIS_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_SHARDINGANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTIR/Analysis/Edge.h"
#include "ttmlir/Dialect/TTIR/Analysis/ShardChainConfig.h"
#include "ttmlir/Dialect/TTIR/Analysis/TTIRAnalysis.h"

namespace mlir::tt::ttir {

enum class ShardingPolicyType {
  DFSharding,
};

struct ShardingAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<LayoutAttr>> legalGrids;
  unsigned usableL1CacheSize = 0;

  ShardingAnalysisInput() : legalGrids() {}

  ShardingAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalGrids,
      unsigned usableL1CacheSize)
      : legalGrids(legalGrids), usableL1CacheSize(usableL1CacheSize) {}

  bool operator==(const ShardingAnalysisInput &rhs) const {
    return legalGrids == rhs.legalGrids;
  }

  bool operator!=(const ShardingAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct ShardingAnalysisResult {
  llvm::DenseMap<Operation *, std::vector<LayoutAttr>> legalGrids;
  std::unordered_set<Edge> reshardedEdges;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule;

  ShardingAnalysisResult() : legalGrids(), reshardedEdges(), schedule() {}

  ShardingAnalysisResult(
      const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalGrids,
      const std::unordered_set<Edge> &reshardedEdges)
      : legalGrids(legalGrids), reshardedEdges(reshardedEdges) {}
};

// Determine shard chain configs.
//
class ShardingAnalysis
    : public TTIRAnalysis<ShardingAnalysisInput, ShardingAnalysisResult> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override;
  std::vector<ShardChainConfig> shardChainConfigs;

public:
  ShardingAnalysis(Operation *op) : TTIRAnalysis(op) {}
};
} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_SHARDINGANALYSIS_H
