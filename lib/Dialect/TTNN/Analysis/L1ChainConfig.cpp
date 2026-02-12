// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"

#include "ttmlir/Dialect/TTNN/Analysis/Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/Analysis/TensorLayouts.h"

#include "llvm/ADT/DenseSet.h"

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
    const TensorTypeLayoutsMap *tensorTypePossibleLayouts,
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
    const llvm::DenseSet<Edge> &overrideReshardEdges,
    const llvm::StringMap<OutputLayoutOverrideParams> &overrideOutputLayout) {
  assert(state == L1ChainState::Built);

  // Reconcile adjacent shard specs.
  // Generate reshard specs where needed.
  //
  ShardSolver shardSolver(tensorTypePossibleLayouts, legalConfigs, opL1MemSpecs,
                          l1ChainedOps, overrideReshardEdges,
                          overrideOutputLayout);

  state = shardSolver.resolve() ? L1ChainState::Resolved : L1ChainState::Failed;

  return shardSolver;
}

void L1ChainConfig::complete(
    const llvm::DenseMap<Operation *, OpConfig> &selectedOpConfig,
    llvm::DenseMap<Edge, MemReconfigEntry> &memReconfigEntryMap) {
  assert(state == L1ChainState::Resolved);
  for (auto &opL1MemSpec : opL1MemSpecs) {
    auto legalConfigsIter = selectedOpConfig.find(opL1MemSpec.op);
    assert(legalConfigsIter != selectedOpConfig.end());

    opL1MemSpec.config = legalConfigsIter->second;
  }

  this->memReconfigEntryMap.swap(memReconfigEntryMap);
  state = L1ChainState::Completed;
}

void L1ChainConfig::complete() {
  assert(state == L1ChainState::Resolved);
  state = L1ChainState::Completed;
}

void L1ChainConfig::merge(L1ChainConfig &other) {
  assert(getState() == other.getState());
  opL1MemSpecs.insert(opL1MemSpecs.end(), other.opL1MemSpecs.begin(),
                      other.opL1MemSpecs.end());
  l1ChainedOps.insert(other.l1ChainedOps.begin(), other.l1ChainedOps.end());
  memReconfigEntryMap.insert(other.memReconfigEntryMap.begin(),
                             other.memReconfigEntryMap.end());
}

} // namespace mlir::tt::ttnn
