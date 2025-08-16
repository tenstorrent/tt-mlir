// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1CHAINCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1CHAINCONFIG_H

#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::tt::ttnn {

struct OpL1MemSpec {
  // Operation that is part of the L1 chain.
  //
  Operation *op;

  // Tensor split factor for the output tensor of the op(working with a partial
  // tensor).
  //
  uint tensorSplitFactor = 1;

  // Op specific configuration.
  //
  OpConfig config;
};

// Enum to track the state of the L1 chain.
// InBuild: L1 chain is being built. OpL1MemSpecs can be added.
// Built: L1 chain is built, but not resolved yet. ShardSolver can be run.
// Resolved: L1 chain is resolved. Reshards are computed. We can pick legal
// configs for each op.
// Completed: L1 chain is completed. OpL1MemSpecs are
// resolved to a single config.
//
enum class L1ChainState { InBuild, Built, Resolved, Completed, Failed };

class L1ChainConfig {
private:
  std::vector<OpL1MemSpec> opL1MemSpecs;
  llvm::DenseSet<Operation *> l1ChainedOps;
  llvm::DenseMap<Edge, MemReconfigEntry> memReconfigEntryMap;
  L1ChainState state = L1ChainState::InBuild;

public:
  L1ChainConfig() : opL1MemSpecs(), state() {}

  ShardSolver resolveWithSolver(
      const TensorTypeLayoutsMap *tensorTypePossibleLayouts,
      const llvm::DenseMap<Operation *, std::vector<OpConfig>> &legalConfigs,
      unsigned usableL1CacheSize, float tensorL1UsageCap,
      const llvm::DenseSet<Edge> &overrideReshardEdges,
      const llvm::StringMap<OutputLayoutOverrideParams> &overrideOutputLayout);
  void resolve();
  void build();
  void complete(const llvm::DenseMap<Operation *, OpConfig> &selectedOpConfig,
                llvm::DenseMap<Edge, MemReconfigEntry> &memReconfigEntryMap);
  void complete();

  bool isEmpty() { return opL1MemSpecs.empty(); }
  void addOpL1MemSpec(OpL1MemSpec spec) {
    assert(state == L1ChainState::InBuild);
    l1ChainedOps.insert(spec.op);
    opL1MemSpecs.push_back(std::move(spec));
  }
  const std::vector<OpL1MemSpec> &getOpL1MemSpecs() const {
    return opL1MemSpecs;
  }
  L1ChainState getState() const { return state; }
  std::string getStateString() const {
    switch (state) {
    case L1ChainState::InBuild:
      return "InBuild";
    case L1ChainState::Built:
      return "Built";
    case L1ChainState::Resolved:
      return "Resolved";
    case L1ChainState::Completed:
      return "Completed";
    case L1ChainState::Failed:
      return "Failed";
    }
  }
  const llvm::DenseMap<Edge, MemReconfigEntry> &getMemReconfigEntryMap() const {
    return memReconfigEntryMap;
  }

  uint64_t size() const { return opL1MemSpecs.size(); }
  void merge(L1ChainConfig &other);

  Operation *getLastOp() const {
    assert(!opL1MemSpecs.empty());
    return opL1MemSpecs.back().op;
  }

  bool spillEndToDRAM = false;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const L1ChainConfig &config) {
  os << "L1ChainConfig(size=" << config.size() << ")";
  os << "\n\tState: " << config.getStateString();
  for (const auto &opL1MemSpec : config.getOpL1MemSpecs()) {
    os << "\n\t";
    opL1MemSpec.op->print(os);
    os << "@ loc: " << utils::getOpLocName(opL1MemSpec.op);
    os << "\n\t\t outputLayout: " << opL1MemSpec.config.outputLayout;
  }
  return os;
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1CHAINCONFIG_H
