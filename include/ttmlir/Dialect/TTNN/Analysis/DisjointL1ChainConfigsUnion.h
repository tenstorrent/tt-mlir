// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_DISJOINTL1CHAINCONFIGSUNION_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_DISJOINTL1CHAINCONFIGSUNION_H

#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"
#include <cstdint>
#include <mlir/IR/Operation.h>

namespace mlir::tt::ttnn {

class DisjoinL1ChainConfigsUnion {
private:
  llvm::DenseMap<Operation *, L1ChainConfig> l1ChainConfigsMap;
  llvm::DenseMap<Operation *, Operation *> parents;

public:
  DisjoinL1ChainConfigsUnion() = default;

  void insertOp(Operation *op);

  /** @return the "representative" node in op's component */
  Operation *findRepresentativeOp(Operation *op);

  /** @return the reference to the l1ChainConfig that containts op*/
  L1ChainConfig &findL1ChainConfig(Operation *op);

  /** @return whether the merge changed connectivity */
  bool mergeChains(Operation *opA, Operation *opB);

  /** @return whether x and y are in the same connected component */
  bool connected(Operation *opA, Operation *opB);

  /** @return the number of disjoint L1ChainConfigs */
  uint64_t size() const;
};

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_DISJOINTL1CHAINCONFIGSUNION_H
