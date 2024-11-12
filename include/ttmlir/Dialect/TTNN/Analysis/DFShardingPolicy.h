// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysisPolicy.h"

namespace mlir::tt::ttnn {

// Process ops in DFS schedulable order and build shard chain configs.
// Schedule is also produced as a side effect of sharding.
//
class DFShardingPolicy : public MemoryLayoutAnalysisPolicy {
private:
  std::unordered_set<Edge> overrideReshardEdges;
  void pickOpShardLayouts(ShardSolver &shardSolver,
                          const L1ChainConfig &l1ChainConfig);

public:
  DFShardingPolicy(
      Operation *rootOp, std::vector<L1ChainConfig> &l1ChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
          &legalLayouts,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : MemoryLayoutAnalysisPolicy(rootOp, l1ChainConfigs, legalLayouts,
                                   schedule, usableL1CacheSize),
        overrideReshardEdges() {}

  void run() final;

  void setOverrideReshardEdges(const std::unordered_set<Edge> &reshardEdges) {
    overrideReshardEdges = reshardEdges;
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H
