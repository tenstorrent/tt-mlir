// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/ShardingAnalysis.h"
#include "ttmlir/Dialect/TTIR/Analysis/DFShardingPolicy.h"

namespace mlir::tt::ttir {

bool ShardingAnalysis::applyOverrides() {

  // TODO(nobradovic):
  // Placeholder, no overrides for now.
  //
  return false;
}

llvm::DenseMap<Operation *, std::vector<LayoutAttr>> filterShardedOnly(
    const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalGrids) {
  llvm::DenseMap<Operation *, std::vector<LayoutAttr>> shardedGrids;
  for (const auto &opGrids : legalGrids) {
    std::vector<LayoutAttr> shardedLayouts;
    for (const auto &layout : opGrids.second) {
      if (layout.hasShardedTensorMemoryLayout()) {
        shardedLayouts.push_back(layout);
      }
    }

    shardedGrids[opGrids.first] = shardedLayouts;
  }

  return shardedGrids;
}

void ShardingAnalysis::analysisImplementation() {
  ShardingPolicyType policy = ShardingPolicyType::DFSharding;

  switch (policy) {
  case ShardingPolicyType::DFSharding:
    DFShardingPolicy dfShardingPolicy(
        op, shardChainConfigs, filterShardedOnly(analysisInput.legalGrids),
        analysisResult.schedule, analysisInput.usableL1CacheSize);
    dfShardingPolicy.run();
    break;
  }

  // Copy over default legal layouts.
  //
  analysisResult.legalGrids = analysisInput.legalGrids;

  // Override with shard chain configs where applicable.
  //
  for (const auto &shardChainConfig : shardChainConfigs) {
    assert(shardChainConfig.getState() == ShardChainState::Completed);
    for (const auto &shardSpec : shardChainConfig.getShardSpecs()) {
      analysisResult.legalGrids[shardSpec.op] =
          std::vector<LayoutAttr>{shardSpec.layout};
    }
  }
}
} // namespace mlir::tt::ttir
