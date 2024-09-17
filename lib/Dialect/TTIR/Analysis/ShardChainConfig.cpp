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
    const llvm::DenseMap<Operation *, std::vector<LayoutAttr>> &legalLayouts,
    unsigned usableL1CacheSize) {
  assert(state == ShardChainState::Built);

  // Reconcile adjacent shard specs.
  // Generate reshard specs where needed.
  //
  ShardSolver shardSolver(legalLayouts, shardSpecs, shardedOps,
                          usableL1CacheSize);
  state = ShardChainState::Resolved;

  return shardSolver;
}

void ShardChainConfig::complete(
    const llvm::DenseMap<Operation *, LayoutAttr> &selectedOpLayout,
    std::unordered_set<Edge> &reshardedEdges) {
  assert(state == ShardChainState::Resolved);
  for (auto &shardSpec : shardSpecs) {
    auto legalLayoutsIter = selectedOpLayout.find(shardSpec.op);
    assert(legalLayoutsIter != selectedOpLayout.end());

    shardSpec.layout = legalLayoutsIter->second;
  }

  this->reshardedEdges.swap(reshardedEdges);
  state = ShardChainState::Completed;
}

} // namespace mlir::tt::ttir
