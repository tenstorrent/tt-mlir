// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardChainConfig.h"
#include <unordered_set>

namespace mlir::tt::ttnn {

// Process ops in DFS schedulable order and build shard chain configs.
// Schedule is also produced as a side effect of sharding.
//
class DFShardingPolicy {
private:
  Operation *rootOp;
  std::vector<ShardChainConfig> *shardChainConfigs;
  llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> legalLayouts;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> *schedule;
  unsigned usableL1CacheSize = 0;

public:
  DFShardingPolicy(
      Operation *rootOp, std::vector<ShardChainConfig> &shardChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
          &legalLayouts,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : rootOp(rootOp), shardChainConfigs(&shardChainConfigs),
        legalLayouts(legalLayouts), schedule(&schedule),
        usableL1CacheSize(usableL1CacheSize) {}

  void run(const std::unordered_set<Edge> &overrideReshardEdges);
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_DFSHARDINGPOLICY_H
