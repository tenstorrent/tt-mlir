// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"

namespace mlir::tt::ttnn {

void L1ChainConfig::build() {
  assert(state == L1ChainState::InBuild);
  state = L1ChainState::Built;
}

void L1ChainConfig::resolve() {
  assert(state == L1ChainState::Built);
  state = L1ChainState::Resolved;
}

ShardSolver L1ChainConfig::resolveWithSolver(
    const llvm::DenseMap<Operation *, std::vector<TensorConfigAttr>>
        &legalLayouts,
    unsigned usableL1CacheSize,
    const std::unordered_set<Edge> &overrideReshardEdges) {
  assert(state == L1ChainState::Built);

  // Reconcile adjacent shard specs.
  // Generate reshard specs where needed.
  //
  ShardSolver shardSolver(legalLayouts, opL1MemSpecs, l1ChainedOps,
                          usableL1CacheSize, overrideReshardEdges);
  state = L1ChainState::Resolved;

  return shardSolver;
}

void L1ChainConfig::complete(
    const llvm::DenseMap<Operation *, TensorConfigAttr> &selectedOpLayout,
    std::unordered_set<Edge> &memReconfigEdges) {
  assert(state == L1ChainState::Resolved);
  for (auto &opL1MemSpec : opL1MemSpecs) {
    auto legalLayoutsIter = selectedOpLayout.find(opL1MemSpec.op);
    assert(legalLayoutsIter != selectedOpLayout.end());

    opL1MemSpec.layout = legalLayoutsIter->second;
  }

  this->memReconfigEdges.swap(memReconfigEdges);
  state = L1ChainState::Completed;
}

} // namespace mlir::tt::ttnn
