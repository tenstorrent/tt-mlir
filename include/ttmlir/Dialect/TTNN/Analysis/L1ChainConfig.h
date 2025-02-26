// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_L1CHAINCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_L1CHAINCONFIG_H

#include "ttmlir/Dialect/TTNN/Analysis/ShardSolver.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

namespace mlir::tt::ttnn {

struct OpL1MemSpec {
  // Operation that is part of the L1 chain.
  //
  Operation *op;

  // Tensor split factor for the output tensor of the op(working with a partial
  // tensor).
  //
  uint tensorSplitFactor = 1;

  // Layout of the output tensor of the op.
  //
  TTNNLayoutAttr layout;
};

// Enum to track the state of the L1 chain.
// InBuild: L1 chain is being built. OpL1MemSpecs can be added.
// Built: L1 chain is built, but not resolved yet. ShardSolver can be run.
// Resolved: L1 chain is resolved. Reshards are computed. We can pick legal
// layouts for each op.
// Completed: L1 chain is completed. OpL1MemSpecs are
// resolved to a single layout.
//
enum class L1ChainState { InBuild, Built, Resolved, Completed, Failed };

class L1ChainConfig {
private:
  std::vector<OpL1MemSpec> opL1MemSpecs;
  llvm::DenseSet<Operation *> l1ChainedOps;
  std::unordered_set<Edge> memReconfigEdges;
  L1ChainState state = L1ChainState::InBuild;
  bool spillEndToDRAM = false;

public:
  L1ChainConfig() : opL1MemSpecs(), state() {}

  ShardSolver resolveWithSolver(
      const llvm::DenseMap<Operation *, std::vector<TTNNLayoutAttr>>
          &legalLayouts,
      unsigned usableL1CacheSize,
      const std::unordered_set<Edge> &overrideReshardEdges);
  void resolve();
  void build();
  void
  complete(const llvm::DenseMap<Operation *, TTNNLayoutAttr> &selectedOpLayout,
           std::unordered_set<Edge> &memReconfigEdges);
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
  const std::unordered_set<Edge> &getMemReconfigEdges() const {
    return memReconfigEdges;
  }

  uint64_t size() const { return opL1MemSpecs.size(); }
  void merge(L1ChainConfig &other);

  Operation *getLastOp() const {
    assert(!opL1MemSpecs.empty());
    return opL1MemSpecs.back().op;
  }

  void setSpillEndToDRAM(bool spill) { spillEndToDRAM = spill; }
  bool shouldSpillEndToDRAM() const { return spillEndToDRAM; }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const L1ChainConfig &config) {
  os << "L1ChainConfig(size=" << config.size() << ")";
  for (const auto &opL1MemSpec : config.getOpL1MemSpecs()) {

    os << "\n\t" << opL1MemSpec.op->getName().getStringRef().str() << "\t"
       << utils::getOpLocName(opL1MemSpec.op);
  }
  return os;
}

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_L1CHAINCONFIG_H
