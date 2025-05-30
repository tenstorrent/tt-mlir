// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_LOOPUTILS_H
#define TTMLIR_LOOPUTILS_H

#include "ttmlir/Dialect/TT/IR/TT.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace ttmlir::utils {

// Returns the outermost possible loop nest level given a set of affine
// load/store index values.
inline mlir::Block *getLoopNestLevel(mlir::ValueRange values) {
  mlir::Region *outermostNestLevel = values[0].getParentRegion();
  for (mlir::Value value : values) {
    mlir::Region *region = value.getParentRegion();
    if (outermostNestLevel->isAncestor(region)) {
      outermostNestLevel = region;
    }
  }
  assert(outermostNestLevel->getBlocks().size() == 1);
  return &outermostNestLevel->front();
}

template <typename OpType>
inline OpType getOutermostLoopNest(mlir::Operation *op) {
  OpType opType = mlir::dyn_cast<OpType>(op);
  OpType maybeOuter = mlir::dyn_cast<OpType>(op->getParentOp());
  while (maybeOuter) {
    opType = maybeOuter;
    maybeOuter = mlir::dyn_cast<OpType>(maybeOuter->getParentOp());
  }
  return opType;
}

template <typename OpType>
inline OpType getOutermostLoopNest(mlir::Region *region) {
  return getOutermostLoopNest<OpType>(region->getParentOp());
}

template <typename OpType>
inline OpType getOutermostLoopNest(mlir::Block *block) {
  return getOutermostLoopNest<OpType>(block->getParent());
}

template <typename OpType>
inline OpType getOutermostLoopNest(mlir::OpOperand &use) {
  return getOutermostLoopNest<OpType>(use.getOwner());
}

template <typename OpType>
inline OpType getOutermostLoopNest(mlir::Value value) {
  return getOutermostLoopNest<OpType>(value.getParentRegion());
}

template <typename OpType>
inline OpType getOutermostLoopNest(mlir::ValueRange values) {
  return getOutermostLoopNest<OpType>(values.front());
}

} // namespace ttmlir::utils

#endif // TTMLIR_LOOPUTILS_H
