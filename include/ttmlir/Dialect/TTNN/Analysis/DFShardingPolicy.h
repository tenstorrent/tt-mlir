// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H

#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"
#include "ttmlir/Dialect/TTNN/Utils/PassOverrides.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttnn {

// Process ops in DFS schedulable order and build shard chain configs.
// Schedule is also produced as a side effect of sharding.
//
class DFShardingPolicy : public MemoryLayoutAnalysisPolicy {
private:
  const TensorTypeLayoutsMap *tensorTypePossibleLayouts;
  llvm::DenseSet<Edge> overrideReshardEdges;
  llvm::StringMap<OutputLayoutOverrideParams> overrideOutputLayout;

  void pickOpShardConfigs(ShardSolver &shardSolver,
                          const L1ChainConfig &l1ChainConfig);

public:
  DFShardingPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const TensorTypeLayoutsMap *tensorTypePossibleLayouts,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule)
      : MemoryLayoutAnalysisPolicy(rootOp, l1ChainConfigs, legalConfigs,
                                   schedule),
        tensorTypePossibleLayouts(tensorTypePossibleLayouts),
        overrideReshardEdges() {}

  void run() final;

  void setOverrides(
      const llvm::DenseSet<Edge> &reshardEdges,
      const llvm::StringMap<OutputLayoutOverrideParams> &overrideOutputLayout) {
    overrideReshardEdges = reshardEdges;
    this->overrideOutputLayout = overrideOutputLayout;
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H
