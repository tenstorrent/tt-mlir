// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTIR_ANALYSIS_SHARDCHAINCONFIG_H
#define TTMLIR_DIALECT_TTIR_ANALYSIS_SHARDCHAINCONFIG_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Analysis/ShardSolver.h"
#include <unordered_set>

namespace mlir::tt::ttir {

struct ShardSpec {
  Operation *op;
  uint tensorSplitFactor;
  LayoutAttr layout;
};

// Enum to track the state of the shard chain.
// InBuild: Shard chain is being built. ShardSpecs can be added.
// Built: Shard chain is built, but not resolved yet. ShardSolver can be run.
// Resolved: Shard chain is resolved. Reshards are computed. We can pick legal
// layouts for each op. Completed: Shard chain is completed. ShardSpecs are
// resolved to a single layout.
//
enum class ShardChainState { InBuild, Built, Resolved, Completed };

class ShardChainConfig {
private:
  std::vector<ShardSpec> shardSpecs;
  llvm::DenseSet<Operation *> shardedOps;
  std::unordered_set<Edge> reshardedEdges;
  ShardChainState state = ShardChainState::InBuild;

public:
  ShardChainConfig() : shardSpecs(), state() {}

  ShardSolver resolve(
      const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalGrids,
      unsigned usableL1CacheSize);
  void build();
  void complete(const llvm::DenseMap<Operation *, LayoutAttr> &selectedOpLayout,
                std::unordered_set<Edge> &reshardedEdges);

  bool isEmpty() { return shardSpecs.empty(); }
  void addShardSpec(ShardSpec &&spec) {
    assert(state == ShardChainState::InBuild);
    shardedOps.insert(spec.op);
    shardSpecs.push_back(std::move(spec));
  }
  const std::vector<ShardSpec> &getShardSpecs() const { return shardSpecs; }
  ShardChainState getState() const { return state; }
};

} // namespace mlir::tt::ttir

#endif // TTMLIR_DIALECT_TTIR_ANALYSIS_SHARDCHAINCONFIG_H
