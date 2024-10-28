// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDINGANALYSIS_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDINGANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TTNNAnalysis.h"

namespace mlir::tt::ttnn {

enum class PolicyType { DFSharding, L1Interleaved };

struct ShardingAnalysisInput {
  llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> legalLayouts;
  unsigned usableL1CacheSize = 0;

  ShardingAnalysisInput() : legalLayouts() {}

  ShardingAnalysisInput(
      const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
          &legalLayouts,
      unsigned usableL1CacheSize)
      : legalLayouts(legalLayouts), usableL1CacheSize(usableL1CacheSize) {}

  bool operator==(const ShardingAnalysisInput &rhs) const {
    return legalLayouts == rhs.legalLayouts;
  }

  bool operator!=(const ShardingAnalysisInput &rhs) const {
    return !(*this == rhs);
  }
};

struct ShardingAnalysisResult {
  llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> legalLayouts;
  std::unordered_set<Edge> reshardedEdges;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> schedule;

  ShardingAnalysisResult() : legalLayouts(), reshardedEdges(), schedule() {}

  ShardingAnalysisResult(
      const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
          &legalLayouts,
      const std::unordered_set<Edge> &reshardedEdges)
      : legalLayouts(legalLayouts), reshardedEdges(reshardedEdges) {}
};

// Determine shard chain configs.
//
class ShardingAnalysis
    : public TTNNAnalysis<ShardingAnalysisInput, ShardingAnalysisResult> {

private:
  void analysisImplementation() override;
  bool applyOverrides() override;
  std::vector<L1ChainConfig> l1ChainConfigs;

public:
  ShardingAnalysis(Operation *op) : TTNNAnalysis(op) {}
};
} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_SHARDINGANALYSIS_H
