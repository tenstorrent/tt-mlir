// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::tt::d2m {

class GenericOp;

/// Compute the CBType and create a d2m.get_cb op for a given operand of a
/// generic op.  Results are cached to ensure each (generic, operand) pair gets
/// exactly one CB value.  Port numbers are assigned sequentially and do NOT
/// correspond to operand indices.
unsigned getCBOperandIdx(GenericOp generic, Value cbGenericOperand);

/// Trace a value through view-like operations (subview, expand_shape, etc.)
/// and return the defining op if it matches OpT.  Returns null otherwise.
template <typename OpT>
OpT traceToDefiningOp(Value value) {
  while (value) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp) {
      return nullptr;
    }
    if (auto op = mlir::dyn_cast<OpT>(definingOp)) {
      return op;
    }
    if (mlir::isa<mlir::ViewLikeOpInterface>(definingOp)) {
      value = definingOp->getOperand(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

} // namespace mlir::tt::d2m

#endif
