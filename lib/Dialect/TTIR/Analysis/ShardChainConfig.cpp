// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/ShardChainConfig.h"
#include "ttmlir/Dialect/TTIR/Analysis/ShardSolver.h"

namespace mlir::tt::ttir {

void ShardChainConfig::build() {
  assert(state == ShardChainState::InBuild);
  state = ShardChainState::Built;
}

ShardSolver ShardChainConfig::resolve(
    const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalGrids,
    unsigned usableL1CacheSize) {
  assert(state == ShardChainState::Built);

  // Reconcile adjacent shard specs.
  // Generate reshard specs where needed.
  //
  ShardSolver shardSolver(legalGrids, shardSpecs, shardedOps,
                          usableL1CacheSize);
  state = ShardChainState::Resolved;

  return shardSolver;
}

void ShardChainConfig::complete(
    const llvm::DenseMap<Operation *, LayoutAttr> &selectedOpLayout,
    std::unordered_set<Edge> &reshardedEdges) {
  assert(state == ShardChainState::Resolved);
  for (auto &shardSpec : shardSpecs) {
    auto legalGridsIter = selectedOpLayout.find(shardSpec.op);
    assert(legalGridsIter != selectedOpLayout.end());

    shardSpec.layout = legalGridsIter->second;
  }

  this->reshardedEdges.swap(reshardedEdges);
  state = ShardChainState::Completed;
}

} // namespace mlir::tt::ttir
