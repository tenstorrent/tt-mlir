// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysis.h"

#include "ttmlir/Dialect/TTNN/Analysis/BFInterleavedPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/DFShardingPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/GreedyL1InterleavedPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

bool MemoryLayoutAnalysis::applyOverrides() {

  // TODO(nobradovic):
  // Placeholder, no overrides for now.
  //
  return false;
}

llvm::DenseMap<Operation *, std::vector<OpConfig>> filterShardedOnly(
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs) {
  llvm::DenseMap<Operation *, std::vector<OpConfig>> shardedConfigs;
  for (const auto &configs : legalConfigs) {
    std::vector<OpConfig> opShardedConfigs;
    for (const OpConfig &config : configs.second) {
      if (config.outputLayout.hasShardedL1TensorMemoryLayout()) {
        opShardedConfigs.push_back(config);
      }
    }

    shardedConfigs[configs.first] = opShardedConfigs;
  }

  return shardedConfigs;
}

llvm::DenseMap<Operation *, std::vector<OpConfig>> filterDRAMAndL1Interleaved(
    const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs) {
  llvm::DenseMap<Operation *, std::vector<OpConfig>> l1InterleavedConfigs;
  for (const auto &configs : legalConfigs) {
    std::vector<OpConfig> opL1InterleavedConfigs;
    for (const auto &config : configs.second) {
      if (config.outputLayout.hasDRAMBufferType() ||
          config.outputLayout.hasInterleavedL1TensorMemoryLayout()) {
        opL1InterleavedConfigs.push_back(config);
      }
    }

    l1InterleavedConfigs[configs.first] = opL1InterleavedConfigs;
  }

  return l1InterleavedConfigs;
}

void MemoryLayoutAnalysis::analysisImplementation() {
  // Apply specific memory layout analysis policy.
  //
  switch (analysisInput.policy) {
  case MemoryLayoutAnalysisPolicyType::DFSharding: {
    DFShardingPolicy dfShardingPolicy(
        op, l1ChainConfigs, analysisInput.tensorTypePossibleLayouts,
        filterShardedOnly(analysisInput.legalConfigs), analysisResult.schedule);
    dfShardingPolicy.setOverrides(analysisInput.overrideReshardEdges,
                                  analysisInput.overrideOutputLayout);
    dfShardingPolicy.run();
    break;
  }
  case MemoryLayoutAnalysisPolicyType::GreedyL1Interleaved: {
    GreedyL1InterleavedPolicy l1InterleavedPolicy(
        op, l1ChainConfigs,
        filterDRAMAndL1Interleaved(analysisInput.legalConfigs),
        analysisResult.schedule);
    l1InterleavedPolicy.run();
    break;
  }
  case MemoryLayoutAnalysisPolicyType::BFInterleaved: {
    BFInterleavedPolicy bfInterleavedPolicy(
        op, l1ChainConfigs,
        filterDRAMAndL1Interleaved(analysisInput.legalConfigs),
        analysisResult.schedule);
    bfInterleavedPolicy.run();
    break;
  }
  }

  // Copy over default legal configs.
  //
  analysisResult.legalConfigs = analysisInput.legalConfigs;

  // Override with L1 chain configs where applicable.
  //
  for (const auto &l1ChainConfig : l1ChainConfigs) {
    if (l1ChainConfig.getState() == L1ChainState::Failed) {
      continue;
    }

    assert(l1ChainConfig.getState() == L1ChainState::Completed);
    for (const auto &opL1MemSpec : l1ChainConfig.getOpL1MemSpecs()) {
      analysisResult.legalConfigs[opL1MemSpec.op] =
          std::vector<OpConfig>{opL1MemSpec.config};
    }

    analysisResult.memReconfigEntryMap.insert(
        l1ChainConfig.getMemReconfigEntryMap().begin(),
        l1ChainConfig.getMemReconfigEntryMap().end());

    // Handle spill location for chain output
    switch (l1ChainConfig.spillLocation) {
    case SpillLocation::DRAM:
      analysisResult.spillToDramOps.push_back(l1ChainConfig.getLastOp());
      break;
    case SpillLocation::L1Interleaved:
      analysisResult.spillToL1InterleavedOps.push_back(
          l1ChainConfig.getLastOp());
      break;
    case SpillLocation::None:
      // No spill needed - chain output consumed directly by next chain
      break;
    }
  }
}
} // namespace mlir::tt::ttnn
