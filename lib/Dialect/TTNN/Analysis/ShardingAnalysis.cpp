// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/ShardingAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/DFShardingPolicy.h"

namespace mlir::tt::ttnn {

bool ShardingAnalysis::applyOverrides() {

  // TODO(nobradovic):
  // Placeholder, no overrides for now.
  //
  return false;
}

llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
filterShardedOnly(const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
                      &legalLayouts) {
  llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> shardedLayouts;
  for (const auto &opLayouts : legalLayouts) {
    std::vector<tt::LayoutAttr> opShardedLayouts;
    for (const auto &layout : opLayouts.second) {
      if (layout.hasShardedL1TensorMemoryLayout()) {
        opShardedLayouts.push_back(layout);
      }
    }

    shardedLayouts[opLayouts.first] = opShardedLayouts;
  }

  return shardedLayouts;
}

void ShardingAnalysis::analysisImplementation() {
  ShardingPolicyType policy = ShardingPolicyType::DFSharding;

  switch (policy) {
  case ShardingPolicyType::DFSharding:
    DFShardingPolicy dfShardingPolicy(
        op, shardChainConfigs, filterShardedOnly(analysisInput.legalLayouts),
        analysisResult.schedule, analysisInput.usableL1CacheSize);
    dfShardingPolicy.run();
    break;
  }

  // Copy over default legal layouts.
  //
  analysisResult.legalLayouts = analysisInput.legalLayouts;

  // Override with shard chain configs where applicable.
  //
  for (const auto &shardChainConfig : shardChainConfigs) {
    assert(shardChainConfig.getState() == ShardChainState::Completed);
    for (const auto &shardSpec : shardChainConfig.getShardSpecs()) {
      analysisResult.legalLayouts[shardSpec.op] =
          std::vector<tt::LayoutAttr>{shardSpec.layout};
    }

    analysisResult.reshardedEdges.insert(
        shardChainConfig.getReshardedEdges().begin(),
        shardChainConfig.getReshardedEdges().end());
  }
}
} // namespace mlir::tt::ttnn
