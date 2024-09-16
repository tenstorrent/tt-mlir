// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_DFSHARDINGPOLICY_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_DFSHARDINGPOLICY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ttmlir/Dialect/TTIR/Analysis/ShardChainConfig.h"

namespace mlir::tt::ttir {

// Process ops in DFS schedulable order and build shard chain configs.
// Schedule is also produced as a side effect of sharding.
//
class DFShardingPolicy {
private:
  Operation *rootOp;
  std::vector<ShardChainConfig> *shardChainConfigs;
  llvm::DenseMap<Operation *, std::vector<LayoutAttr>> legalLayouts;
  llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> *schedule;
  unsigned usableL1CacheSize = 0;

public:
  DFShardingPolicy(
      Operation *rootOp, std::vector<ShardChainConfig> &shardChainConfigs,
      const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalLayouts,
      llvm::DenseMap<func::FuncOp, llvm::SmallVector<Operation *>> &schedule,
      unsigned usableL1CacheSize)
      : rootOp(rootOp), shardChainConfigs(&shardChainConfigs),
        legalLayouts(legalLayouts), schedule(&schedule),
        usableL1CacheSize(usableL1CacheSize) {}

  void run();
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_DFSHARDINGPOLICY_H
