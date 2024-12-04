// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/DisjointL1ChainConfigsUnion.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"

namespace mlir::tt::ttnn {

void DisjoinL1ChainConfigsUnion::insertOp(Operation *op) {
  assert(parents.count(op) == 0);
  parents[op] = op;
  l1ChainConfigsMap[op] = L1ChainConfig();
}

Operation *DisjoinL1ChainConfigsUnion::findRepresentativeOp(Operation *op) {
  if (parents[op] == op) {
    return op;
  }

  parents[op] = findRepresentativeOp(parents[op]);
  return parents[op];
}

L1ChainConfig &DisjoinL1ChainConfigsUnion::findL1ChainConfig(Operation *op) {
  return l1ChainConfigsMap[findRepresentativeOp(op)];
}

bool DisjoinL1ChainConfigsUnion::mergeChains(Operation *opa, Operation *opb) {
  Operation *opa_root = findRepresentativeOp(opa);
  Operation *opb_root = findRepresentativeOp(opb);
  if (opa_root == opb_root) {
    return false;
  }

  L1ChainConfig &l1ChainConfigA = findL1ChainConfig(opa_root);
  L1ChainConfig &l1ChainConfigB = findL1ChainConfig(opb_root);
  if (l1ChainConfigA.size() < l1ChainConfigB.size()) {
    std::swap(opa_root, opb_root);
  }

  l1ChainConfigA.merge(l1ChainConfigB);
  l1ChainConfigsMap.erase(opb_root);
  parents[opb_root] = opa_root;

  return true;
}

bool DisjoinL1ChainConfigsUnion::connected(Operation *opA, Operation *opB) {
  return findRepresentativeOp(opA) == findRepresentativeOp(opB);
}

uint64_t DisjoinL1ChainConfigsUnion::size() const {
  return l1ChainConfigsMap.size();
}

} // namespace mlir::tt::ttnn
