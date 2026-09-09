// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_CBUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tt::d2m {

class GenericOp;

Value getOrCreateCB(RewriterBase &rewriter, GenericOp generic, Block *block,
                    unsigned cbOperandIndex);

/// Optional op attribute mapping logical generic operand indices to physical CB
/// ports. Consumers default to identity when the attribute is absent.
StringRef getPhysicalCBPortMapAttrName();
void setPhysicalCBPortMap(Operation *op, ArrayRef<int64_t> logicalToPhysical);
DenseI64ArrayAttr getPhysicalCBPortMap(Operation *op);
int64_t getPhysicalCBPort(Operation *op, int64_t logicalOperandIdx);

/// Reuse CB ports for additional CB-backed generic operands whose lifetimes do
/// not overlap. Input/output operands are treated as fixed ports and count
/// against `maxPhysicalCBPorts`, but are not remapped.
void reuseDisjointCBPorts(RewriterBase &rewriter, GenericOp generic,
                          int64_t maxPhysicalCBPorts = 32);

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
