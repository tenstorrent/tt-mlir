// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_DISJOINTL1CHAINCONFIGSUNION_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_DISJOINTL1CHAINCONFIGSUNION_H

#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>

namespace mlir::tt::ttnn {

class DisjoinL1ChainConfigsUnion {
private:
  llvm::DenseMap<Operation *, L1ChainConfig> l1ChainConfigsMap;
  llvm::DenseMap<Operation *, Operation *> parents;

public:
  DisjoinL1ChainConfigsUnion() = default;

  // Inserts OpL1MemSpec in the same L1ChainConfig that op belongs to.
  // In case the op is not provided, a new L1ChainConfig is created.
  void insertOpL1MemSpec(OpL1MemSpec opL1MemSpec, Operation *referenceOp);

  // Inserts new L1ChainConfig in the union and construct
  // the parent tree for its ops.
  void insertL1ChainConfig(L1ChainConfig &l1ChainConfig);

  /** @return the "representative" op in op's component */
  Operation *findRepresentativeOp(Operation *op);

  /** @return the reference to the l1ChainConfig that containts op */
  L1ChainConfig &findL1ChainConfig(Operation *op);

  /** @return the "representative" op of a newly merged L1ChainConfig */
  Operation *mergeL1ChainConfigs(Operation *opA, Operation *opB);

  /** @return whether x and y are in the same connected component */
  bool connected(Operation *opA, Operation *opB);

  /** @return the number of disjoint L1ChainConfigs */
  uint64_t getNumberOfL1Chains();

  /** @return the number of ops in the chain that contains op */
  uint64_t getNumberOfOpsInChain(Operation *op);

  llvm::DenseMap<Operation *, L1ChainConfig> &getL1ChainConfigs() {
    return l1ChainConfigsMap;
  }
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_DISJOINTL1CHAINCONFIGSUNION_H
