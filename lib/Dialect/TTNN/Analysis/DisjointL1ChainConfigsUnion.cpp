// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/DisjointL1ChainConfigsUnion.h"
#include "ttmlir/Dialect/TTNN/Analysis/L1ChainConfig.h"

namespace mlir::tt::ttnn {

void DisjoinL1ChainConfigsUnion::insertOpL1MemSpec(OpL1MemSpec opL1MemSpec,
                                                   Operation *referenceOp) {
  if (referenceOp == nullptr) {
    L1ChainConfig l1ChainConfig;
    l1ChainConfig.addOpL1MemSpec(opL1MemSpec);
    insertL1ChainConfig(l1ChainConfig);
  } else {
    parents[opL1MemSpec.op] = findRepresentativeOp(referenceOp);
    findL1ChainConfig(referenceOp).addOpL1MemSpec(opL1MemSpec);
  }
}

void DisjoinL1ChainConfigsUnion::insertL1ChainConfig(
    L1ChainConfig &l1ChainConfig) {
  assert(!l1ChainConfig.isEmpty());

  // Construct parent tree for new l1ChainConfig
  Operation *parentOp = l1ChainConfig.getOpL1MemSpecs()[0].op;
  for (auto &opL1MemSpec : l1ChainConfig.getOpL1MemSpecs()) {
    parents[opL1MemSpec.op] = parentOp;
  }

  l1ChainConfigsMap[parentOp] = l1ChainConfig;
}

Operation *DisjoinL1ChainConfigsUnion::findRepresentativeOp(Operation *op) {
  if (op == nullptr) {
    return nullptr;
  }

  if (!parents.count(op)) {
    return nullptr;
  }

  if (parents[op] == op) {
    return op;
  }

  parents[op] = findRepresentativeOp(parents[op]);
  return parents[op];
}

L1ChainConfig &DisjoinL1ChainConfigsUnion::findL1ChainConfig(Operation *op) {
  return l1ChainConfigsMap[findRepresentativeOp(op)];
}

Operation *DisjoinL1ChainConfigsUnion::mergeL1ChainConfigs(Operation *opA,
                                                           Operation *opB) {
  Operation *opA_root = findRepresentativeOp(opA);
  Operation *opB_root = findRepresentativeOp(opB);

  if (opA_root == nullptr) {
    return opB_root;
  }

  if (opB_root == nullptr) {
    return opA_root;
  }

  if (opA_root == opB_root) {
    return opA_root;
  }

  L1ChainConfig &l1ChainConfigA = findL1ChainConfig(opA_root);
  L1ChainConfig &l1ChainConfigB = findL1ChainConfig(opB_root);
  if (l1ChainConfigA.size() >= l1ChainConfigB.size()) {
    l1ChainConfigA.merge(l1ChainConfigB);
  } else {
    l1ChainConfigB.merge(l1ChainConfigA);
    std::swap(opA_root, opB_root);
  }

  l1ChainConfigsMap.erase(opB_root);
  parents[opB_root] = opA_root;

  return opA_root;
}

bool DisjoinL1ChainConfigsUnion::connected(Operation *opA, Operation *opB) {
  return findRepresentativeOp(opA) == findRepresentativeOp(opB);
}

uint64_t DisjoinL1ChainConfigsUnion::getNumberOfL1Chains() {
  return l1ChainConfigsMap.size();
}

uint64_t DisjoinL1ChainConfigsUnion::getNumberOfOpsInChain(Operation *op) {
  return findL1ChainConfig(op).size();
}

} // namespace mlir::tt::ttnn
