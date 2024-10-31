// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/MemoryLayoutAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/DFShardingPolicy.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1InterleavedPolicy.h"

namespace mlir::tt::ttnn {

bool MemoryLayoutAnalysis::applyOverrides() {

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

llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
filterL1InterleavedOnly(
    const llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>>
        &legalLayouts) {
  llvm::DenseMap<Operation *, std::vector<tt::LayoutAttr>> l1InterleavedLayouts;
  for (const auto &opLayouts : legalLayouts) {
    std::vector<tt::LayoutAttr> opL1InterleavedLayouts;
    for (const auto &layout : opLayouts.second) {
      if (layout.hasInterleavedL1TensorMemoryLayout()) {
        opL1InterleavedLayouts.push_back(layout);
      }
    }

    l1InterleavedLayouts[opLayouts.first] = opL1InterleavedLayouts;
  }

  return l1InterleavedLayouts;
}

void MemoryLayoutAnalysis::analysisImplementation() {
  // Apply specific memory layout analysis policy.
  //
  switch (analysisInput.policy) {
  case MemoryLayoutAnalysisPolicyType::DFSharding: {
    DFShardingPolicy dfShardingPolicy(
        op, l1ChainConfigs, filterShardedOnly(analysisInput.legalLayouts),
        analysisResult.schedule, analysisInput.usableL1CacheSize);
    dfShardingPolicy.run(analysisInput.overrideReshardEdges);
    break;
  }
  case MemoryLayoutAnalysisPolicyType::L1Interleaved: {
    L1InterleavedPolicy l1InterleavedPolicy(
        op, l1ChainConfigs, filterL1InterleavedOnly(analysisInput.legalLayouts),
        analysisResult.schedule, analysisInput.usableL1CacheSize);
    l1InterleavedPolicy.run(analysisInput.overrideReshardEdges);
    break;
  }
  }

  // Copy over default legal layouts.
  //
  analysisResult.legalLayouts = analysisInput.legalLayouts;

  // Override with L1 chain configs where applicable.
  //
  for (const auto &l1ChainConfig : l1ChainConfigs) {
    assert(l1ChainConfig.getState() == L1ChainState::Completed);
    for (const auto &opL1MemSpec : l1ChainConfig.getOpL1MemSpecs()) {
      analysisResult.legalLayouts[opL1MemSpec.op] =
          std::vector<tt::LayoutAttr>{opL1MemSpec.layout};
    }

    analysisResult.memReconfigEdges.insert(
        l1ChainConfig.getMemReconfigEdges().begin(),
        l1ChainConfig.getMemReconfigEdges().end());
  }
}
} // namespace mlir::tt::ttnn
